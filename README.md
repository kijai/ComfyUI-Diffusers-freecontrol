# ComfyUI wrapper node for the original Freecontrol diffusers implementation

## WORK IN PROGRESS

default pca_info from original repo: https://drive.google.com/file/d/1o1BcIBANukeJ2pCG064-eNH9hbQoB24Z/view?usp=sharing
goes into `ComfyUI/models/pca_info``

Note: Inversion latents are cached in this custom nodes folder, each takes 30-100MB. Clearing the cache is up to you for now.

Requires xformers, tested with pytorch 2.2.0 and xformers 0.0.24