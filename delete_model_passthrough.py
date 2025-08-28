import torch
import gc
import psutil
import os
import ctypes

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


class DeleteModelPassthrough:
    """
    ComfyUI Custom Node that:
      - Accepts any input (data) and a MODEL (`model`).
      - Deletes the MODEL completely (VRAM + RAM).
      - Passes through the data unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (any_typ,),
                "model": (any_typ,),
            },
        }
 
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = "Memory Management"

    def run(self, data, model):

        before = psutil.virtual_memory().percent
        collected = gc.collect()
        after = psutil.virtual_memory().percent
        freedr = before - after
        print(f"System RAM freed: {freedr:.2f}%, GC collected {collected} objects.")

        if not torch.cuda.is_available():
            return "CUDA not available."
        initial = torch.cuda.memory_allocated()
        
        try:
            del model
            torch.cuda.empty_cache()
            gc.collect()
            print("Deleted MODEL completely from VRAM.")
        except Exception as e:
            print(f"Failed to delete model: {e}")

        final = torch.cuda.memory_allocated()
        freed = max(0, initial - final)
        print(f"GPU VRAM freed: {freed/1e9:.2f} GB")

        return (data,)


NODE_CLASS_MAPPINGS = {
    "DeleteModelPassthrough": DeleteModelPassthrough,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeleteModelPassthrough": "Delete Model (Passthrough Any)",
}
