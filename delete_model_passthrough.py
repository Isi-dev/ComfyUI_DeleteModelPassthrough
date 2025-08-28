import torch, gc, psutil
import comfy.model_management as mm

class AnyType(str):
    def __ne__(self, __value: object) -> bool: return False
any_typ = AnyType("*")


def hard_free(model):
    try:
        # Explicitly nuke parameters and buffers
        if hasattr(model, "parameters"):
            for p in model.parameters():
                if p is not None:
                    p.detach_()
                    del p
        if hasattr(model, "buffers"):
            for b in model.buffers():
                if b is not None:
                    del b
        # Clear state_dict if exists
        if hasattr(model, "state_dict"):
            sd = model.state_dict()
            for k in list(sd.keys()):
                del sd[k]
            del sd
    except Exception as e:
        print(f"Hard free error: {e}")

    try:
        del model
    except:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def purge_specific_from_comfy(model):
    """
    Surgically remove THIS model object from ComfyUI's caches
    without touching unrelated ones. Returns a type string.
    """
    deleted_type = "Unknown"

    try:
        if mm.current_model == model:
            mm.current_model = None
            deleted_type = "Diffusion/UNet"

        if hasattr(mm, "loaded_models"):
            for k, v in list(mm.loaded_models.items()):
                if v is model:
                    del mm.loaded_models[k]
                    deleted_type = "Diffusion/UNet"

        if hasattr(mm, "loaded_clip"):
            for k, v in list(mm.loaded_clip.items()):
                if v is model:
                    del mm.loaded_clip[k]
                    deleted_type = "CLIP"

        if hasattr(mm, "loaded_vae"):
            for k, v in list(mm.loaded_vae.items()):
                if v is model:
                    del mm.loaded_vae[k]
                    deleted_type = "VAE"

    except Exception as e:
        print(f"Cache purge error: {e}")

    return deleted_type


class DeleteModelPassthrough:
    """
    ComfyUI Custom Node:
      - Accepts any input + a model (UNet/CLIP/VAE/latent).
      - Deletes ONLY that model from VRAM, RAM, and ComfyUI cache.
      - Auto-detects the model type and logs it.
      - Passes the data unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"data": (any_typ,), "model": (any_typ,)}}
 
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = "Memory Management"

    def run(self, data, model):
        before_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        before_res  = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        before_ram  = psutil.virtual_memory().percent

        deleted_type = purge_specific_from_comfy(model)
        hard_free(model)

        after_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        after_res  = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        after_ram  = psutil.virtual_memory().percent

        print(f"🗑️ Deleted model type: {deleted_type}")
        print(f"System RAM change: {before_ram - after_ram:.2f}%")
        print(f"GPU allocated freed: {(before_vram - after_vram)/1e9:.2f} GB")
        print(f"GPU reserved freed: {(before_res - after_res)/1e9:.2f} GB")

        return (data,)


NODE_CLASS_MAPPINGS = {"DeleteModelPassthrough": DeleteModelPassthrough}
NODE_DISPLAY_NAME_MAPPINGS = {"DeleteModelPassthrough": "Delete Model (Passthrough Any)"}
