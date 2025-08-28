# ComfyUI_DeleteModelPassthrough
A memory management custom node that removes a given model completely from VRAM after use, while passing through any other input (IMAGE, LATENT, CLIP, STRING, INT, CONDITIONING, VAE, etc.) unchanged. This helps free memory in low-RAM/VRAM environments and reduces the risk of out of memory errors in long workflows.
