import torch
import argparse
import random
import re
import numpy as np
from typing import List, Optional, Union
from .utils import setup_logging

import torch.nn.functional as F
#from sklearn.cluster import KMeans
import warnings

setup_logging()
import logging

logger = logging.getLogger(__name__)


def prepare_scheduler_for_custom_training(noise_scheduler, device):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)


def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler):
    # fix beta: zero terminal SNR
    logger.info(f"fix noise scheduler betas: https://arxiv.org/abs/2305.08891")

    def enforce_zero_terminal_snr(betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas

    betas = noise_scheduler.betas
    betas = enforce_zero_terminal_snr(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # logger.info(f"original: {noise_scheduler.betas}")
    # logger.info(f"fixed: {betas}")

    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alphas_cumprod = alphas_cumprod


def apply_snr_weight(loss, timesteps, noise_scheduler, gamma, v_prediction=False):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    loss = loss * snr_weight
    return loss


def scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler):
    scale = get_snr_scale(timesteps, noise_scheduler)
    loss = loss * scale
    return loss


def get_snr_scale(timesteps, noise_scheduler):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
    scale = snr_t / (snr_t + 1)
    # # show debug info
    # logger.info(f"timesteps: {timesteps}, snr_t: {snr_t}, scale: {scale}")
    return scale


def add_v_prediction_like_loss(loss, timesteps, noise_scheduler, v_pred_like_loss):
    scale = get_snr_scale(timesteps, noise_scheduler)
    # logger.info(f"add v-prediction like loss: {v_pred_like_loss}, scale: {scale}, loss: {loss}, time: {timesteps}")
    loss = loss + loss / scale * v_pred_like_loss
    return loss


def apply_debiased_estimation(loss, timesteps, noise_scheduler, debias_limit_val = 0):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
    weight = 1 / torch.sqrt(snr_t)
    
    if debias_limit_val > 0:
        weight = torch.minimum(weight, torch.ones_like(weight) * debias_limit_val)
    
    loss = weight * loss
    return loss


# TODO train_utilと分散しているのでどちらかに寄せる


def add_custom_train_arguments(parser: argparse.ArgumentParser, support_weighted_captions: bool = True):
    parser.add_argument(
        "--min_snr_gamma",
        type=float,
        default=None,
        help="gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by paper. / 低いタイムステップでの高いlossに対して重みを減らすためのgamma値、低いほど効果が強く、論文では5が推奨",
    )
    parser.add_argument(
        "--scale_v_pred_loss_like_noise_pred",
        action="store_true",
        help="scale v-prediction loss like noise prediction loss / v-prediction lossをnoise prediction lossと同じようにスケーリングする",
    )
    parser.add_argument(
        "--v_pred_like_loss",
        type=float,
        default=None,
        help="add v-prediction like loss multiplied by this value / v-prediction lossをこの値をかけたものをlossに加算する",
    )
    parser.add_argument(
        "--debiased_estimation_loss",
        action="store_true",
        help="debiased estimation loss / debiased estimation loss",
    )
    if support_weighted_captions:
        parser.add_argument(
            "--weighted_captions",
            action="store_true",
            default=False,
            help="Enable weighted captions in the standard style (token:1.3). No commas inside parens, or shuffle/dropout may break the decoder. / 「[token]」、「(token)」「(token:1.3)」のような重み付きキャプションを有効にする。カンマを括弧内に入れるとシャッフルやdropoutで重みづけがおかしくなるので注意",
        )


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(tokenizer, prompt: List[str], max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    tokenizer,
    text_encoder,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    eos: int,
    pad: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            if pad == eos:  # v1
                text_input_chunk[:, -1] = text_input[0, -1]
            else:  # v2
                for j in range(len(text_input_chunk)):
                    if text_input_chunk[j, -1] != eos and text_input_chunk[j, -1] != pad:  # 最後に普通の文字がある
                        text_input_chunk[j, -1] = eos
                    if text_input_chunk[j, 1] == pad:  # BOSだけであとはPAD
                        text_input_chunk[j, 1] = eos

            if clip_skip is None or clip_skip == 1:
                text_embedding = text_encoder(text_input_chunk)[0]
            else:
                enc_out = text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
                text_embedding = enc_out["hidden_states"][-clip_skip]
                text_embedding = text_encoder.text_model.final_layer_norm(text_embedding)

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        if clip_skip is None or clip_skip == 1:
            text_embeddings = text_encoder(text_input)[0]
        else:
            enc_out = text_encoder(text_input, output_hidden_states=True, return_dict=True)
            text_embeddings = enc_out["hidden_states"][-clip_skip]
            text_embeddings = text_encoder.text_model.final_layer_norm(text_embeddings)
    return text_embeddings


def get_weighted_text_embeddings(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    clip_skip=None,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    prompt_tokens, prompt_weights = get_prompts_with_weights(tokenizer, prompt, max_length - 2)

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        no_boseos_middle=no_boseos_middle,
        chunk_length=tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        tokenizer,
        text_encoder,
        prompt_tokens,
        tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=device)

    # assign weights to the prompts and normalize in the sense of mean
    previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings = text_embeddings * prompt_weights.unsqueeze(-1)
    current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings = text_embeddings * (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    return text_embeddings


# https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2
def pyramid_noise_like(noise, device, iterations=6, discount=0.4):
    b, c, w, h = noise.shape  # EDIT: w and h get over-written, rename for a different variant!
    u = torch.nn.Upsample(size=(w, h), mode="bilinear").to(device)
    for i in range(iterations):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        wn, hn = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += u(torch.randn(b, c, wn, hn).to(device)) * discount**i
        if wn == 1 or hn == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


# https://www.crosslabs.org//blog/diffusion-with-offset-noise
def apply_noise_offset(latents, noise, noise_offset, adaptive_noise_scale):
    if noise_offset is None:
        return noise
    if adaptive_noise_scale is not None:
        # latent shape: (batch_size, channels, height, width)
        # abs mean value for each channel
        latent_mean = torch.abs(latents.mean(dim=(2, 3), keepdim=True))

        # multiply adaptive noise scale to the mean value and add it to the noise offset
        noise_offset = noise_offset + adaptive_noise_scale * latent_mean
        noise_offset = torch.clamp(noise_offset, 0.0, None)  # in case of adaptive noise scale is negative

    noise = noise + noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
    return noise


def apply_masked_loss(loss, batch):
    # mask image is -1 to 1. we need to convert it to 0 to 1
    mask_image = batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)  # use R channel

    # resize to the same size as the loss
    mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
    mask_image = mask_image / 2 + 0.5
    loss = loss * mask_image
    return loss


"""
##########################################
# Perlin Noise
def rand_perlin_2d(device, shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (
        torch.stack(
            torch.meshgrid(torch.arange(0, res[0], delta[0], device=device), torch.arange(0, res[1], delta[1], device=device)),
            dim=-1,
        )
        % 1
    )
    angles = 2 * torch.pi * torch.rand(res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = (
        lambda slice1, slice2: gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )
    dot = lambda grad, shift: (
        torch.stack((grid[: shape[0], : shape[1], 0] + shift[0], grid[: shape[0], : shape[1], 1] + shift[1]), dim=-1)
        * grad[: shape[0], : shape[1]]
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return 1.414 * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(device, shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape, device=device)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(device, shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def perlin_noise(noise, device, octaves):
    _, c, w, h = noise.shape
    perlin = lambda: rand_perlin_2d_octaves(device, (w, h), (4, 4), octaves)
    noise_perlin = []
    for _ in range(c):
        noise_perlin.append(perlin())
    noise_perlin = torch.stack(noise_perlin).unsqueeze(0)   # (1, c, w, h)
    noise += noise_perlin # broadcast for each batch
    return noise / noise.std()  # Scaled back to roughly unit variance
"""



def adjust_losses(all_losses, penalty=0.05):
    # penalty can be interperted as the limit of how much penalization can be applied, penalty * loss <- approaches as deviation gets larger
    highest_loss = all_losses.max() 
    lowest_loss = all_losses.min()  
    average_loss = all_losses.mean()

    range_to_highest = highest_loss - average_loss
    range_to_lowest = average_loss - lowest_loss
    adjusted_losses = torch.zeros_like(all_losses)

    for i, loss in enumerate(all_losses):
        if loss > average_loss: # high loss
            fraction_above = (loss - average_loss) / (range_to_highest + 1e-6)
            adjust_down = penalty * fraction_above # maybe change to l2 loss, do fraction_above^2
            adjusted_losses[i] = loss * (1 - adjust_down)
            #adjusted_losses[i] = average_loss + (loss-average_loss)*(1-adjust_down)
            
        else: # low loss
            fraction_below = (average_loss - loss) / (range_to_lowest + 1e-6) # positive val
            adjust_up = penalty * fraction_below
            adjusted_losses[i] = loss * (1 + adjust_up)
            #adjusted_losses[i] = average_loss - (average_loss-loss)*(1-adjust_up) # changed plus to minus, to make it go towards the loss

    adjusted_losses *= all_losses.sum() / adjusted_losses.sum()

    return adjusted_losses

def adjust_losses_min_sq_error(all_losses):
    highest_loss = all_losses.max() 
    lowest_loss = all_losses.min()  
    average_loss = all_losses.mean()

    range_to_highest = highest_loss - average_loss
    range_to_lowest = average_loss - lowest_loss
    adjusted_losses = torch.zeros_like(all_losses)
    
    for i, loss in enumerate(all_losses):
        if loss > average_loss: # high loss
            avg_diff = loss - average_loss
            fraction_above = avg_diff / (range_to_highest + 1e-6)
            
            # from 0.5, and 0.05
            
            if fraction_above > 0.7 or False: # large deviation from mean 
                # the min(0.5, error) makes small deviations do 
                #  for this if case, 1 - error*min(0.5,) >=  1-error^2, make hard cap of penatly 50% of difference
                # adjustment, meaning a 30% deviation from mean makes a
                adjust_down = min(0.7, fraction_above)* fraction_above # maybe change to l2 loss, do fraction_above^2
            else: # small deviation from mean
                adjust_down = min(0.10, fraction_above)* fraction_above
            adjusted_losses[i] = loss * (1 - adjust_down)
            #adjusted_losses[i] = average_loss + (avg_diff)*(1-adjust_down)
            
        else: # low loss
            avg_diff = average_loss - loss
            fraction_below = (avg_diff) / (range_to_lowest + 1e-6)
            if fraction_below > 0.7 or False: # large deviation from mean
                adjust_up = min(0.7, fraction_below)* fraction_below # maybe change to l2 loss, do fraction_above^2
            else: # small deviation from mean
                adjust_up = min(0.10, fraction_below)* fraction_below
            adjusted_losses[i] = loss * (1 + adjust_up)
            #adjusted_losses[i] = average_loss - (avg_diff)*(1-adjust_up)
    
    max_difference_ratio = 25 #  highest sample probability can only reach 25 x lowest sample probability
    new_max = adjusted_losses.max()
    new_min = adjusted_losses.min()
    if new_max/new_min > max_difference_ratio:
        print("max ratio hit")
        # Calculate the desired maximum value
        desired_max_val = max_difference_ratio * new_min
        # Calculate the scaling factor
        scaling_factor = desired_max_val / new_max

        # Scale the array, I should do adjusted_losses - new_min to properly scale them, but the effects are the same and this method
        # can reduce the number of times it access this function cause it scales it to be <50 and not exactly 50x in the worst case
        adjusted_losses = (adjusted_losses * scaling_factor)+new_min
        
    # scaling area to be 1
    adjusted_losses *= all_losses.sum() / adjusted_losses.sum()
    return adjusted_losses

def timestep_attention(loss_map, b_size, device, max_timesteps=1000, default_loss=0.2):#run_number,
    #global adjusted_probabilities_print
    stat_dict = {}
    if loss_map:
        #all_timesteps = range(1,  max_timesteps)
        #all_losses = torch.tensor([loss_map.get(t, default_loss) for t in all_timesteps], device=device) # check this 
        
        all_timesteps = torch.arange(1, max_timesteps, dtype=torch.long, device=device)
        all_losses = torch.tensor([loss_map.get(t.item(), default_loss) for t in all_timesteps], device=device) # check this

        # Adjust the losses based on the specified criteria
        #adjusted_losses = adjust_losses(all_losses)
        
        #adjusted_losses = adjust_losses_min_sq_error(all_losses)
        #for t_i, l_i in zip(all_timesteps, adjusted_losses.tolist()):
        #    loss_map[t_i] = l_i
        
        adjusted_losses = adjust_losses_min_sq_error(all_losses)
        
        # Calculate new probabilities with the adjusted losses
        adjusted_probabilities = adjusted_losses / adjusted_losses.sum()

        sampled_indices = torch.multinomial(adjusted_probabilities, b_size, replacement=True)
        skewed_timesteps = all_timesteps[sampled_indices]
        
        
        loss_map_val = adjusted_probabilities.tolist()
        min_prob = min(loss_map_val)
        max_prob = max(loss_map_val)
        area_under_curve = [np.sum(loss_map_val[:200]),
                            np.sum(loss_map_val[200:400]),
                            np.sum(loss_map_val[400:600]),
                            np.sum(loss_map_val[600:800]),
                            np.sum(loss_map_val[800:]),                       
                           ]
        stat_dict = {"loss_map/min_scaled":min_prob,
                     "loss_map/max_scaled":max_prob,
                     "loss_map/Prob_minmax_multiplier":  max_prob/min_prob,
                     
                     "AuC/area [0-20]": area_under_curve[0],
                     "AuC/area [20-40]": area_under_curve[1],
                     "AuC/area [40-60]": area_under_curve[2],
                     "AuC/area [60-80]": area_under_curve[3],
                     "AuC/area [80-100]": area_under_curve[4]
                    }
        
        
    else: # ignore this case (wasabi)
        # Generate log-normal samples for timesteps as the fallback
        mean, std = 0.00, 1.00
        lognorm_samples = torch.distributions.LogNormal(mean, std).sample((b_size,)).to(device)
        normalized_samples = lognorm_samples / lognorm_samples.max()
        skewed_timesteps = (normalized_samples * (max_timesteps - 1)).long()

    # Log the adjusted probabilities
    #dir_name = f"H:\\TimestepAttention\\run{run_number}"
    #if not os.path.exists(dir_name):
    #    os.makedirs(dir_name)
    # List existing files and find the next available file number
    #existing_files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f)) and 'run_probabilities_' in f]
    #highest_num = 0
    #for file_name in existing_files:
    #    parts = file_name.replace('run_probabilities_', '').split('.')
    #    try:
    #        num = int(parts[0])
    #        if num > highest_num:
    #            highest_num = num
    #    except ValueError:
            # This handles the case where the file name does not end with a number
    #        continue

    # Determine the filename for the new log
    #new_file_num = highest_num + 1
    #file_out = os.path.join(dir_name, f"run_probabilities_{new_file_num}.txt")

    #adjusted_probabilities_print = adjusted_probabilities.cpu().tolist()
    #timesteps_probs_str = ', '.join(map(str, adjusted_probabilities_print))
    #with open(file_out, 'w') as file:
    #    file.write(timesteps_probs_str + '\n')
    
    
    
    return skewed_timesteps, stat_dict

def update_loss_map_ema(loss_map, new_losses, timesteps, max_timesteps=1000, default_loss=0.2, update_fraction=0.2, dtype=torch.float32):#update_fraction=0.5
    if not isinstance(new_losses, torch.Tensor):
        new_losses = torch.tensor(new_losses, dtype=dtype)
    warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.tensor.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.utils.checkpoint*")
    timesteps = torch.tensor(timesteps, dtype=torch.long, device=new_losses.device)
    # Initialize all timesteps with a basic loss value of 0.1 if they are not already present

    # just reshape loss and get the list
    aggregated_losses = new_losses.view(-1).tolist()
    
    # Define decay weights for adjusting adjacent timesteps
    #decay_weights = [0.01, 0.03, 0.09, 0.25, 0.35, 0.50] # error potentially?
    decay_weights = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05]
    if len(aggregated_losses) != len(timesteps):
        print("loss and timestep doesn't match")
    
    for i, timestep_value in enumerate(timesteps.tolist()):
        loss_value = aggregated_losses[i] if i < len(aggregated_losses) else 0.0

        # Correctly handle the definition of current_average_loss
        # This ensures it's always defined before being used
        current_average_loss = loss_map.get(timestep_value, default_loss)

        # Directly update the timestep with the specified update fraction
        loss_difference = (loss_value - current_average_loss)
        weighted_difference =  loss_difference * update_fraction
        
        new_average_loss = current_average_loss + weighted_difference
        loss_map[timestep_value] = new_average_loss

        # Apply decayed adjustments for adjacent timesteps
        for offset, weight in enumerate(decay_weights, start=1):
            for direction in [-1, 1]:
                adjacent_timestep = timestep_value + direction * offset
                if 1 <= adjacent_timestep < 1000:  # Ensure it's within the valid range
                    if adjacent_timestep in loss_map: # unnecessary check
                        adjacent_current_loss = loss_map[adjacent_timestep]
                        # Calculate the decayed loss value based on distance and main loss value
                        decayed_loss_value = loss_value -  loss_difference * weight
                        # Apply update fraction to move towards the decayed loss value
                        new_adjacent_loss = adjacent_current_loss + (decayed_loss_value - adjacent_current_loss) * update_fraction
                        loss_map[adjacent_timestep] = new_adjacent_loss
    
    #all_timesteps = torch.arange(1, max_timesteps, dtype=torch.long, device=device)
    #  max_timesteps=1000,
    return loss_map

def update_loss_map_ema_edit(loss_map, new_losses, timesteps,  max_timesteps=1000, update_fraction=0.2, default_loss=0.2, dtype=torch.float32):
    if not isinstance(new_losses, torch.Tensor):
        new_losses = torch.tensor(new_losses, dtype=dtype)
    warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.tensor.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.utils.checkpoint*")
    timesteps = torch.tensor(timesteps, dtype=torch.long, device=new_losses.device)
    # Initialize all timesteps with a basic loss value of 0.1 if they are not already present

    # just reshape
    aggregated_losses = new_losses.view(-1).tolist()
    if len(aggregated_losses) != len(timesteps):
        print("loss and timestep doesn't match")
    # Define decay weights for adjusting adjacent timesteps
    #decay_weights = [0.01, 0.03, 0.09, 0.25, 0.35, 0.50] # error potentially?
    decay_weights = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05]
    for i, timestep_value in enumerate(timesteps.tolist()):
        loss_value = aggregated_losses[i] if i < len(aggregated_losses) else 0.0

        # Correctly handle the definition of current_average_loss
        # This ensures it's always defined before being used
        current_average_loss = loss_map.get(timestep_value, default_loss)

        # Directly update the timestep with the specified update fraction
        loss_difference = (loss_value - current_average_loss)
        weighted_difference =  loss_difference * update_fraction
        
        new_average_loss = current_average_loss + weighted_difference
        loss_map[timestep_value] = new_average_loss

        # Apply decayed adjustments for adjacent timesteps
        for offset, weight in enumerate(decay_weights, start=1):
            for direction in [-1, 1]:
                adjacent_timestep = timestep_value + direction * offset
                if 1 <= adjacent_timestep < 1000:  # Ensure it's within the valid range
                    if adjacent_timestep in loss_map: # unnecessary check
                        adjacent_current_loss = loss_map[adjacent_timestep]
                        # Calculate the decayed loss value based on distance and main loss value
                        
                        # removed: bc ur doing decayed_error diff * 0.5
                        #decayed_loss_value = loss_value -  loss_difference * weight
                        # Apply update fraction to move towards the decayed loss value
                        #new_adjacent_loss = adjacent_current_loss + (decayed_loss_value - adjacent_current_loss) * update_fraction
                        #modded:
                        new_adjacent_loss = adjacent_current_loss + (loss_value - adjacent_current_loss) * weight
                        
                        loss_map[adjacent_timestep] = new_adjacent_loss
    
    return loss_map
