# ComfyUI_DeleteModelPassthrough
A memory management custom node that removes a given model completely from VRAM after use, while passing through any other input (IMAGE, LATENT, CLIP, STRING, INT, CONDITIONING, VAE, etc.) unchanged. This helps free memory in low-RAM & VRAM environments and reduces the risk of out of memory errors in long workflows.

## 📌 Overview
This custom node provides a **memory management utility** for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).  
It allows you to **delete a specific model** (checkpoint, etc.) completely from **VRAM and system RAM** after use, while **passing through any other input type unchanged** (IMAGE, LATENT, CLIP, STRING, INT, CONDITIONING, VAE, etc.).  

This is especially useful for **low VRAM & low RAM environments**, helping to reduce *out-of-memory (OOM) errors* in long workflows.

---

## ⚙️ Node Details
- **Name:** `Delete Model (Passthrough Any)`  
- **Inputs:**
  - `data` → any input type (IMAGE, LATENT, CLIP, STRING, INT, CONDITIONING, VAE, etc.)
  - `model` → the MODEL you want to remove
- **Outputs:**
  - The `data` input, passed through unchanged
- **Effect:**  
  Deletes the `model` completely from Python, VRAM, and RAM using:
  ```python
  del model
  torch.cuda.empty_cache()
  gc.collect()


🛠️ Installation

1. Navigate to your ComfyUI custom_nodes folder:
  ```python
  cd .../ComfyUI/custom_nodes
  ```

2. Clone or copy this repository into the folder:
  ```python
  git clone https://github.com/Isi-dev/ComfyUI_DeleteModelPassthrough.git
  ```
3. Install dependencies:
  ```python
  pip install -r requirements.txt
  ```

📝 Usage

Assume you have a large CLIP model that you want to remove from VRAM (without unloading it into low system RAM) before loading your diffusion model to avoid OOM errors:

- Connect the output from the CLIPTextEncode node into this node’s data input.

- Connect the output from the CLIPLoader node into this node’s model input.

- Connect the output from this node into your sampler node.

The CLIP model will be deleted from memory after use, while your encoded text (data) continues downstream into the workflow.

