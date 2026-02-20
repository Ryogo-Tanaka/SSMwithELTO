import subprocess
import torch

# Cached to ensure selection runs only once per process
_cached_device: torch.device | None = None
_cached_all: list[torch.device] | None = None

def select_device(prefer_memory: bool = True) -> torch.device:
    """Select the best available torch.device.

    When prefer_memory=True, picks the GPU with the most free memory via nvidia-smi.
    """
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        _cached_device = torch.device("cpu")
        return _cached_device

    if prefer_memory:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
                encoding="utf-8"
            )
            free_memories = [int(line) for line in output.strip().splitlines()]
            best_index = max(range(len(free_memories)), key=lambda i: free_memories[i])
            print(f"Selecting GPU:{best_index} (free memory: {free_memories[best_index]} MiB)")
            _cached_device = torch.device(f"cuda:{best_index}")
        except Exception as e:
            print(f"nvidia-smi query failed ({e}); falling back to cuda:0")
            _cached_device = torch.device("cuda:0")
    else:
        print("Using default GPU:0")
        _cached_device = torch.device("cuda:0")

    return _cached_device

def list_available_devices(prefer_memory: bool = True) -> list[torch.device]:
    """Return all available devices, optionally sorted by free memory descending."""
    global _cached_all
    if _cached_all is not None:
        return _cached_all

    if not torch.cuda.is_available():
        _cached_all = [torch.device("cpu")]
        return _cached_all

    n = torch.cuda.device_count()
    if not prefer_memory:
        _cached_all = [torch.device(f"cuda:{i}") for i in range(n)]
    else:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,nounits,noheader"],
                encoding="utf-8"
            )
            free_memories = [int(l) for l in out.strip().splitlines()]
            sorted_idx = sorted(range(n), key=lambda i: free_memories[i], reverse=True)
            _cached_all = [torch.device(f"cuda:{i}") for i in sorted_idx]
        except Exception:
            _cached_all = [torch.device(f"cuda:{i}") for i in range(n)]
    return _cached_all