import torch, gc, psutil
import comfy.model_management as mm
from comfy.model_management import current_loaded_models, LoadedModel

class AnyType(str):
    def __ne__(self, __value: object) -> bool: return False
any_typ = AnyType("*")


def hard_free(model):
    """
    Forcefully free model memory by detaching parameters and clearing caches.
    """
    try:
        # Clear parameters and buffers
        if hasattr(model, "parameters"):
            for p in model.parameters():
                if p is not None:
                    p.detach_()
                    if hasattr(p, 'data'):
                        del p.data
                    del p
        
        if hasattr(model, "buffers"):
            for b in model.buffers():
                if b is not None:
                    b.detach_()
                    if hasattr(b, 'data'):
                        del b.data
                    del b
        
        # Clear any cached attributes
        for attr_name in list(vars(model).keys()):
            attr = getattr(model, attr_name)
            if isinstance(attr, torch.Tensor):
                attr.detach_()
                delattr(model, attr_name)
                
    except Exception as e:
        print(f"Hard free error: {e}")

    try:
        del model
    except:
        pass

    gc.collect()
    mm.soft_empty_cache()


def purge_specific_from_comfy(model_obj):
    """
    Remove a specific model from ComfyUI's current_loaded_models list.
    Returns the model type if found and removed.
    """
    deleted_type = "Unknown"
    
    try:
        # Check if it's a LoadedModel wrapper
        if isinstance(model_obj, LoadedModel):
            if model_obj in current_loaded_models:
                current_loaded_models.remove(model_obj)
                # Also detach the underlying model
                if hasattr(model_obj, 'model_unload'):
                    model_obj.model_unload()
                deleted_type = f"LoadedModel ({type(model_obj.model).__name__})"
        
        # Check if it's a raw model that might be wrapped in LoadedModel
        else:
            for i, loaded_model in enumerate(current_loaded_models):
                if loaded_model.model is model_obj:
                    current_loaded_models.pop(i)
                    if hasattr(loaded_model, 'model_unload'):
                        loaded_model.model_unload()
                    deleted_type = f"RawModel ({type(model_obj).__name__})"
                    break
        
        # Additional cleanup for common model types
        if hasattr(model_obj, '__class__'):
            cls_name = model_obj.__class__.__name__.lower()
            if 'unet' in cls_name:
                deleted_type = "UNet"
            elif 'clip' in cls_name:
                deleted_type = "CLIP"
            elif 'vae' in cls_name:
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
        if model is None:
            print("⚠️ No model provided to delete")
            return (data,)
            
        # Get memory stats before deletion
        before_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        before_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        before_ram = psutil.virtual_memory().percent

        # Remove from ComfyUI management and free memory
        deleted_type = purge_specific_from_comfy(model)
        hard_free(model)

        # Get memory stats after deletion
        after_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        after_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        after_ram = psutil.virtual_memory().percent

        # Log results
        print(f"🗑️ Deleted model type: {deleted_type}")
        print(f"System RAM change: {before_ram - after_ram:.2f}%")
        
        if torch.cuda.is_available():
            vram_freed = (before_vram - after_vram) / (1024 * 1024 * 1024)
            reserved_freed = (before_reserved - after_reserved) / (1024 * 1024 * 1024)
            print(f"GPU allocated freed: {vram_freed:.3f} GB")
            print(f"GPU reserved freed: {reserved_freed:.3f} GB")

        return (data,)


# Node mappings
NODE_CLASS_MAPPINGS = {"DeleteModelPassthrough": DeleteModelPassthrough}
NODE_DISPLAY_NAME_MAPPINGS = {"DeleteModelPassthrough": "Delete Model (Passthrough Any)"}
