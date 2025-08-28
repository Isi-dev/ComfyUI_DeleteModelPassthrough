import torch
import gc
import psutil
import os
import ctypes

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
                "data": (any,),
                "model": (any,),
            },
        }
 
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = "Memory Management"

    def _free_gpu_vram(self):
        if not torch.cuda.is_available():
            return "CUDA not available."

        initial = torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        final = torch.cuda.memory_allocated()
        freed = max(0, initial - final)
        return f"GPU VRAM freed: {freed/1e9:.2f} GB"

    def _free_system_ram(self):
        before = psutil.virtual_memory().percent
        collected = gc.collect()
        after = psutil.virtual_memory().percent
        freed = before - after
        return f"System RAM freed: {freed:.2f}%, GC collected {collected} objects."

    def run(self, data, model):
        logs = []

        try:
            del model
            torch.cuda.empty_cache()
            gc.collect()
            logs.append("Deleted MODEL completely from VRAM and RAM.")
        except Exception as e:
            logs.append(f"Failed to delete model: {e}")

        logs.append(self._free_gpu_vram())
        logs.append(self._free_system_ram())

        return (data,)


NODE_CLASS_MAPPINGS = {
    "DeleteModelPassthrough": DeleteModelPassthrough,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeleteModelPassthrough": "Delete Model (Passthrough Any)",
}
