# import torch, gc, psutil
# import comfy.model_management as mm
# from comfy.model_management import current_loaded_models, LoadedModel

# class AnyType(str):
#     def __ne__(self, __value: object) -> bool: return False
# any_typ = AnyType("*")


# def hard_free(model):
#     """
#     Forcefully free model memory by detaching parameters and clearing caches.
#     """
#     try:
#         # Clear parameters and buffers
#         if hasattr(model, "parameters"):
#             for p in model.parameters():
#                 if p is not None:
#                     p.detach_()
#                     if hasattr(p, 'data'):
#                         del p.data
#                     del p
        
#         if hasattr(model, "buffers"):
#             for b in model.buffers():
#                 if b is not None:
#                     b.detach_()
#                     if hasattr(b, 'data'):
#                         del b.data
#                     del b
        
#         # Clear any cached attributes
#         for attr_name in list(vars(model).keys()):
#             attr = getattr(model, attr_name)
#             if isinstance(attr, torch.Tensor):
#                 attr.detach_()
#                 delattr(model, attr_name)
                
#     except Exception as e:
#         print(f"Hard free error: {e}")

#     try:
#         del model
#     except:
#         pass

#     gc.collect()
#     mm.soft_empty_cache()


# def purge_specific_from_comfy(model_obj):
#     """
#     Remove a specific model from ComfyUI's current_loaded_models list.
#     Returns the model type if found and removed.
#     """
#     deleted_type = "Unknown"
    
#     try:
#         # Check if it's a LoadedModel wrapper
#         if isinstance(model_obj, LoadedModel):
#             if model_obj in current_loaded_models:
#                 current_loaded_models.remove(model_obj)
#                 # Also detach the underlying model
#                 if hasattr(model_obj, 'model_unload'):
#                     model_obj.model_unload()
#                 deleted_type = f"LoadedModel ({type(model_obj.model).__name__})"
        
#         # Check if it's a raw model that might be wrapped in LoadedModel
#         else:
#             for i, loaded_model in enumerate(current_loaded_models):
#                 if loaded_model.model is model_obj:
#                     current_loaded_models.pop(i)
#                     if hasattr(loaded_model, 'model_unload'):
#                         loaded_model.model_unload()
#                     deleted_type = f"RawModel ({type(model_obj).__name__})"
#                     break
        
#         # Additional cleanup for common model types
#         if hasattr(model_obj, '__class__'):
#             cls_name = model_obj.__class__.__name__.lower()
#             if 'unet' in cls_name:
#                 deleted_type = "UNet"
#             elif 'clip' in cls_name:
#                 deleted_type = "CLIP"
#             elif 'vae' in cls_name:
#                 deleted_type = "VAE"
                
#     except Exception as e:
#         print(f"Cache purge error: {e}")
    
#     return deleted_type


# class DeleteModelPassthrough:
#     """
#     ComfyUI Custom Node:
#       - Accepts any input + a model (UNet/CLIP/VAE/latent).
#       - Deletes ONLY that model from VRAM, RAM, and ComfyUI cache.
#       - Auto-detects the model type and logs it.
#       - Passes the data unchanged.
#     """

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {"required": {"data": (any_typ,), "model": (any_typ,)}}
 
#     RETURN_TYPES = (any_typ,)
#     RETURN_NAMES = ("output",)
#     FUNCTION = "run"
#     CATEGORY = "Memory Management"

#     def run(self, data, model):
#         if model is None:
#             print("⚠️ No model provided to delete")
#             return (data,)
            
#         # Get memory stats before deletion
#         before_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
#         before_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
#         before_ram = psutil.virtual_memory().percent

#         # Remove from ComfyUI management and free memory
#         deleted_type = purge_specific_from_comfy(model)
#         hard_free(model)

#         # Get memory stats after deletion
#         after_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
#         after_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
#         after_ram = psutil.virtual_memory().percent

#         # Log results
#         print(f"🗑️ Deleted model type: {deleted_type}")
#         print(f"System RAM change: {before_ram - after_ram:.2f}%")
        
#         if torch.cuda.is_available():
#             vram_freed = (before_vram - after_vram) / (1024 * 1024 * 1024)
#             reserved_freed = (before_reserved - after_reserved) / (1024 * 1024 * 1024)
#             print(f"GPU allocated freed: {vram_freed:.3f} GB")
#             print(f"GPU reserved freed: {reserved_freed:.3f} GB")

#         return (data,)


# # Node mappings
# NODE_CLASS_MAPPINGS = {"DeleteModelPassthrough": DeleteModelPassthrough}
# NODE_DISPLAY_NAME_MAPPINGS = {"DeleteModelPassthrough": "Delete Model (Passthrough Any)"}


# Testing Deepseek's more rigorous version

import torch, gc, psutil
import comfy.model_management as mm
from comfy.model_management import current_loaded_models, LoadedModel

class AnyType(str):
    def __ne__(self, __value: object) -> bool: return False
any_typ = AnyType("*")


def hard_free(model):
    """
    Forcefully free model memory by detaching parameters and clearing caches.
    Special handling for different model types.
    """
    if model is None:
        return
        
    try:
        model_type = type(model).__name__
        print(f"🔧 Freeing {model_type} model...")
        
        # Handle ModelPatcher objects (common for CLIP, UNet, VAE)
        if model_type == 'ModelPatcher' and hasattr(model, 'model'):
            print("   Detaching ModelPatcher parameters...")
            real_model = model.model
            if hasattr(real_model, "parameters"):
                for p in real_model.parameters():
                    if p is not None:
                        p.detach_()
                        del p
            model.model = None
        
        # Handle ControlNet models (they have different structure)
        elif 'Control' in model_type or 'control' in model_type.lower():
            print("   Detaching ControlNet parameters...")
            if hasattr(model, "parameters"):
                for p in model.parameters():
                    if p is not None:
                        p.detach_()
                        del p
            if hasattr(model, "weights"):
                model.weights = None
            if hasattr(model, "model"):
                model.model = None
        
        # Standard cleanup for any model
        if hasattr(model, "parameters"):
            for p in model.parameters():
                if p is not None:
                    p.detach_()
                    del p
        
        if hasattr(model, "buffers"):
            for b in model.buffers():
                if b is not None:
                    b.detach_()
                    del b
        
        # Clear any cached tensor attributes
        for attr_name in list(vars(model).keys()):
            attr = getattr(model, attr_name)
            if isinstance(attr, torch.Tensor):
                attr.detach_()
                delattr(model, attr_name)
                
    except Exception as e:
        print(f"❌ Hard free error: {e}")

    try:
        del model
    except:
        pass

    gc.collect()
    mm.soft_empty_cache()


def print_tracked_models():
    """Print all models currently tracked by ComfyUI's main system"""
    print("📋 Models in current_loaded_models:")
    if not current_loaded_models:
        print("   No models tracked in main system")
        return 0
    
    count = 0
    for i, loaded_model in enumerate(current_loaded_models):
        try:
            if hasattr(loaded_model, 'model'):
                model_obj = loaded_model.model
                model_type = type(model_obj).__name__
                model_id = id(model_obj)
                print(f"   {i}: {model_type} (id: {model_id})")
                count += 1
        except Exception as e:
            print(f"   {i}: [Error: {e}]")
    
    return count


def identify_model_type(model_obj):
    """Identify what type of model this is"""
    if model_obj is None:
        return "Unknown"
    
    cls_name = model_obj.__class__.__name__.lower()
    
    if 'clip' in cls_name:
        return "CLIP"
    elif 'unet' in cls_name:
        return "UNet"
    elif 'vae' in cls_name:
        return "VAE"
    elif 'control' in cls_name:
        return "ControlNet"
    elif 'modelpatcher' in cls_name:
        # Check what's inside the ModelPatcher
        if hasattr(model_obj, 'model') and model_obj.model is not None:
            inner_type = model_obj.model.__class__.__name__.lower()
            if 'clip' in inner_type:
                return "CLIP (ModelPatcher)"
            elif 'unet' in inner_type:
                return "UNet (ModelPatcher)"
            elif 'vae' in inner_type:
                return "VAE (ModelPatcher)"
        return "ModelPatcher"
    elif 'diffusion' in cls_name:
        return "DiffusionModel"
    else:
        return f"Unknown ({model_obj.__class__.__name__})"


def purge_specific_from_comfy(model_obj):
    """
    Try to remove model from ComfyUI's tracking systems.
    Returns the identified model type.
    """
    if model_obj is None:
        return "None"
    
    model_type = identify_model_type(model_obj)
    model_removed = False
    
    print(f"🔍 Target model type: {model_type}")
    print(f"🔍 Model object type: {type(model_obj).__name__}")
    print(f"🔍 Model ID: {id(model_obj)}")
    
    # Print current tracking state
    initial_count = print_tracked_models()
    
    try:
        # Try to remove from main tracking system (for CLIP/UNet/VAE)
        if current_loaded_models:
            for i, loaded_model in enumerate(current_loaded_models):
                try:
                    # Check if this LoadedModel contains our target model
                    if (hasattr(loaded_model, 'model') and 
                        (loaded_model.model is model_obj or 
                         (hasattr(model_obj, 'model') and loaded_model.model is model_obj.model))):
                        
                        print(f"🗑️ Removing from tracked models: {type(loaded_model.model).__name__}")
                        current_loaded_models.pop(i)
                        if hasattr(loaded_model, 'model_unload'):
                            loaded_model.model_unload()
                        model_removed = True
                        break
                except:
                    continue
        
        if not model_removed:
            print("⚠️ Model not in main tracking system - may use different management")
            
            # For ControlNet and other specially managed models, we can't remove them from tracking
            # but we can still free their memory
            if 'control' in model_type.lower():
                print("💡 ControlNet models often use custom management systems")
            
    except Exception as e:
        print(f"❌ Error during purge attempt: {e}")
    
    # Print final tracking state
    final_count = print_tracked_models()
    print(f"📊 Tracking change: {initial_count} → {final_count} models")
    
    return model_type


class DeleteModelPassthrough:
    """
    ComfyUI Custom Node: Deletes models from memory regardless of tracking system.
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

        print("=" * 60)
        print("🧹 Starting model deletion process")
        print("=" * 60)
        
        # Try to remove from tracking and free memory
        deleted_type = purge_specific_from_comfy(model)
        hard_free(model)

        # Get memory stats after deletion
        after_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        after_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        after_ram = psutil.virtual_memory().percent

        # Log results
        # print("=" * 60)
        # print("📊 Memory Free Results:")
        # print("=" * 60)
        # print(f"🗑️ Deleted model type: {deleted_type}")
        # print(f"💾 System RAM change: {before_ram - after_ram:+.2f}%")
        
        if torch.cuda.is_available():
            vram_freed = (before_vram - after_vram) / (1024 * 1024 * 1024)
            reserved_freed = (before_reserved - after_reserved) / (1024 * 1024 * 1024)
            # print(f"🎮 GPU allocated freed: {vram_freed:.3f} GB")
            # print(f"🎮 GPU reserved freed: {reserved_freed:.3f} GB")
            # print(f"🎮 Final allocated: {after_vram / (1024 * 1024 * 1024):.3f} GB")
            # print(f"🎮 Final reserved: {after_reserved / (1024 * 1024 * 1024):.3f} GB")
        
        print("=" * 60)
        
        # Success determination based on memory freed
        if torch.cuda.is_available() and reserved_freed > 1:  # Freed at least 1000MB
            print("✅ SUCCESS: Significant memory freed!")
        elif torch.cuda.is_available() and reserved_freed > 0:
            print("⚠️ PARTIAL: Some memory freed, but model may still be referenced")
        else:
            print("❌ FAILED: No significant memory freed - model may be already freed or invalid")
        
        print("=" * 60)

        return (data,)


# Node mappings
NODE_CLASS_MAPPINGS = {"DeleteModelPassthrough": DeleteModelPassthrough}
NODE_DISPLAY_NAME_MAPPINGS = {"DeleteModelPassthrough": "Delete Model (Passthrough Any)"}
