import subprocess
import torch

# デバイス選択結果をキャッシュ
_cached_device: torch.device | None = None
_cached_all: list[torch.device] | None = None

def select_device(prefer_memory: bool = True) -> torch.device:
    """
    最適なtorch.deviceを選択

    1) CUDA利用不可 → CPU
    2) prefer_memory=True → 空きメモリ最大のGPU
    3) prefer_memory=False → cuda:0
    """
    global _cached_device
    # キャッシュ済みならそれを返す
    if _cached_device is not None:
        return _cached_device

    # CUDA利用不可 → CPU
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        _cached_device = torch.device("cpu")
        return _cached_device

    # CUDA利用可
    if prefer_memory:
        try:
            # nvidia-smiで空きメモリ取得
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
                encoding="utf-8"
            )
            free_memories = [int(line) for line in output.strip().splitlines()]
            # 空きメモリ最大のGPUを選択
            best_index = max(range(len(free_memories)), key=lambda i: free_memories[i])
            print(f"Selecting GPU:{best_index} (free memory: {free_memories[best_index]} MiB)")
            _cached_device = torch.device(f"cuda:{best_index}")
        except Exception as e:
            # 失敗時はcuda:0にフォールバック
            print(f"nvidia-smi query failed ({e}); falling back to cuda:0")
            _cached_device = torch.device("cuda:0")
    else:
        # デフォルトでGPU 0を使用
        print("Using default GPU:0")
        _cached_device = torch.device("cuda:0")

    return _cached_device

def list_available_devices(prefer_memory: bool = True) -> list[torch.device]:
    """
    利用可能なデバイスリストを返す
    - CUDA無し: [cpu]
    - prefer_memory=True: 空きメモリ降順でGPUをソート
    - prefer_memory=False: [cuda:0, cuda:1, ...]
    """
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
            # 空きメモリ降順でソート
            sorted_idx = sorted(range(n), key=lambda i: free_memories[i], reverse=True)
            _cached_all = [torch.device(f"cuda:{i}") for i in sorted_idx]
        except Exception:
            _cached_all = [torch.device(f"cuda:{i}") for i in range(n)]
    return _cached_all
