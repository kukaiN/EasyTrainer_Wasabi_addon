## EasyTrainer_Wasabi_addon
 Small changes to the kohya backend and Easytrainer frontend that adds features to easy trainer.

 You will need to install Lora_EasyTrainer first. (I recommend the dev branch, instructions below). If you already have a working install, Maybe makee a separate folder (at least for now, 7/10/24).

This is the specific version of EasyTrainer I am using (it's the latest version available, when I'm writing this, 7/10/24).
https://github.com/derrian-distro/LoRA_Easy_Training_Scripts/tree/80e2c49


# Instructions:

1. Download Derrian's Easy trainer to your perfered location:
   Go to your perferred location, type cmd in the folder address bar, get the latest dev version with:
```
git clone -b dev https://github.com/derrian-distro/LoRA_Easy_Training_Scripts
```
 Then do the basic installing for Derrian EasyTrainer with "install.bat".

2. Then come back to this repo, download the zip or git clone this repo anywhere u like:
```
  git clone https://github.com/kukaiN/EasyTrainer_Wasabi_addon
```
3. go into the downloaded repo and copy everything in the changes folder. Then paste it inside the Easytrainer folder. It'll ask you if you want to replace ~7 files, and once you replaced the files the files are properly updated to the same one I'm using.

4. Then you should have a new file "install_matplotlib.bat" in the easytrainer folder, click it to additionally install matplotlib in the Kohya backend.

Instalization is complete, you start the app normally with the "run.bat"

If everything goes as planned, you should see a new section under the optimizer args tab, like the image below:

![Screenshot 2024-07-10 161417_raw](https://github.com/kukaiN/EasyTrainer_Wasabi_addon/assets/50426885/a7f6f634-ce4f-418a-9f22-14aa5d859386)


## Added/modded Features:
### Minor Changes
  - Better Wandb logging
### Under the Optimizer tab
  - Time Attention mechanism
  - Debiased Estimation loss
    - Max constraint on Debiased Estimation
  - Turn off/on TE2 training


# Acknowledgements:
Time Attention is a modification from here: https://github.com/Anzhc/Timestep-Attention-and-other-shenanigans
