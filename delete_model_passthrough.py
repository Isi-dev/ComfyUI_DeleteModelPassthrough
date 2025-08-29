import torch, gc, psutil
import comfy.model_management as mm
from comfy.model_management import loaded_models, free_memory, get_torch_device

class AnyType(str):
    def __ne__(self, __value: object) -> bool: return False
any_typ = AnyType("*")


def hard_free_model(model):
        
    if model is None:
        return
        
    try:
        model_type = type(model).__name__
        print(f"🔧 Freeing {model_type} model...")
        
        # Handle dictionary-style models (common in some workflows)
        if isinstance(model, dict):
            # print("📦 Handling dictionary-style model container...")
            for key, value in list(model.items()):
                if hasattr(value, 'parameters') or hasattr(value, 'model'):
                    # print(f"   Freeing {key}: {type(value).__name__}")
                    hard_free_model(value)
                del model[key]
            return
        
        # Handle ModelPatcher objects
        if hasattr(model, 'model') and model.model is not None:
            # print("   Detaching inner model from ModelPatcher...")
            inner_model = model.model
            if hasattr(inner_model, "parameters"):
                for p in inner_model.parameters():
                    if p is not None:
                        p.detach_()
                        del p
            model.model = None
        
        # Standard parameter cleanup
        if hasattr(model, "parameters"):
            for p in model.parameters():
                if p is not None:
                    p.detach_()
                    del p
        
        # Clear buffers and tensor attributes
        if hasattr(model, "buffers"):
            for b in model.buffers():
                if b is not None:
                    b.detach_()
                    del b
        
        for attr_name in list(vars(model).keys()):
            attr = getattr(model, attr_name)
            if isinstance(attr, torch.Tensor):
                attr.detach_()
                delattr(model, attr_name)
                
    except Exception as e:
        print(f"❌ Error during model freeing: {e}")


def identify_model_type(model_obj):
    """Identify what type of model this is"""
    if model_obj is None:
        return "Unknown"
    
    if isinstance(model_obj, dict):
        model_types = []
        for key, value in model_obj.items():
            if hasattr(value, '__class__'):
                model_types.append(f"{key}:{value.__class__.__name__}")
        return f"DictContainer[{', '.join(model_types)}]"
    
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
        return "ModelPatcher"
    elif 'diffusion' in cls_name:
        return "DiffusionModel"
    else:
        return f"Unknown ({model_obj.__class__.__name__})"


def print_currently_loaded():
    """Print models currently loaded in ComfyUI's management system"""
    current_models = mm.loaded_models()
    print("📋 Models in loaded_models():")
    if not current_models:
        print("   No models currently managed")
        return 0
    
    for i, model in enumerate(current_models):
        try:
            model_type = identify_model_type(model)
            memory_used = model.model_memory() / (1024 * 1024) if hasattr(model, 'model_memory') else "unknown"
            print(f"   {i}: {model_type} ({memory_used:.1f} MB)")
        except Exception as e:
            print(f"   {i}: [Error: {e}]")
    
    return len(current_models)


class DeleteModelPassthrough:
    """
    ComfyUI Custom Node: Properly deletes models using ComfyUI's memory management system
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


        
        model_type = identify_model_type(model)
        # print(f"🔍 Target model type: {model_type}")
        
        # Print current state
        initial_count = print_currently_loaded()
        
        # Try to remove from ComfyUI's management system
        model_removed = False
        current_models = mm.loaded_models()
        
        if model in current_models:
            # print("🗑️ Removing model from ComfyUI management...")
            current_models.remove(model)
            model_removed = True
        else:
            # Check if it's a ModelPatcher or wrapped model
            for managed_model in current_models:
                try:
                    if (hasattr(managed_model, 'model') and 
                        (managed_model.model is model or 
                         (hasattr(model, 'model') and managed_model.model is model.model))):
                        # print("🗑️ Removing wrapped model from ComfyUI management...")
                        current_models.remove(managed_model)
                        model_removed = True
                        break
                except:
                    continue
        
        # Free memory using ComfyUI's proper methods
        # print("💾 Freeing memory using ComfyUI's system...")
        mm.free_memory(1e30, mm.get_torch_device(), mm.loaded_models())
        
        # Additional forceful cleanup
        # print("🧹 Forceful cleanup...")
        hard_free_model(model)
        
        # ComfyUI's cache cleanup
        mm.soft_empty_cache(force=True)
        
        # Standard Python cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Get memory stats after deletion
        after_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        after_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        after_ram = psutil.virtual_memory().percent

        # Print final state
        final_count = print_currently_loaded()
        
        print(f"📊 Managed models: {initial_count} → {final_count}")
        print(f"💾 System RAM change: {before_ram - after_ram:+.2f}%")
        
        if torch.cuda.is_available():
            vram_freed = (before_vram - after_vram) / (1024 * 1024 * 1024)
            reserved_freed = (before_reserved - after_reserved) / (1024 * 1024 * 1024)
            print(f"GPU allocated freed: {vram_freed:.3f} GB")
            print(f"GPU reserved freed: {reserved_freed:.3f} GB")
            # print(f"Final allocated: {after_vram / (1024 * 1024 * 1024):.3f} GB")
            # print(f"Final reserved: {after_reserved / (1024 * 1024 * 1024):.3f} GB")
        
        print("=" * 60)
        
        # Success determination
        if model_removed:
            print("SUCCESS: Model removed from management system!")
        elif torch.cuda.is_available() and reserved_freed > 0.1:
            print("SUCCESS: Significant memory freed (model may use custom management)")
        else:
            print("⚠PARTIAL: Model may still be referenced somewhere")
        
        print("=" * 60)

        return (data,)



# Node mappings
NODE_CLASS_MAPPINGS = {"DeleteModelPassthrough": DeleteModelPassthrough}
NODE_DISPLAY_NAME_MAPPINGS = {"DeleteModelPassthrough": "Delete Model (Passthrough Any)"}
