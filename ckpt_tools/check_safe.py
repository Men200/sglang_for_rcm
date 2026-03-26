import torch


FILTER_KEYWORD = "patch_embedding"
PT_PATH_A = "/share/workspace/aigc/rcm/assets/Wan2.2-I2V-A14B-high-rCM4.0-merged.pt"
PT_PATH_B = "/share/rcm/models/Wan2.2-I2V-A14B/Wan2.2-I2V-A14B-high.pth"


def _load_state_dict(pt_path: str) -> dict:
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def _extract_filtered_tensors(state_dict: dict) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in state_dict.items():
        if FILTER_KEYWORD in key and torch.is_tensor(value):
            result[key] = value
    return result


def _compare_filtered_tensors(path_a: str, path_b: str) -> None:
    sd_a = _load_state_dict(path_a)
    sd_b = _load_state_dict(path_b)

    tensors_a = _extract_filtered_tensors(sd_a)
    tensors_b = _extract_filtered_tensors(sd_b)

    print(f"\n===== Compare keyword: {FILTER_KEYWORD} =====")
    print(f"A: {path_a} (matched={len(tensors_a)})")
    print(f"B: {path_b} (matched={len(tensors_b)})")

    common_keys = sorted(set(tensors_a.keys()) & set(tensors_b.keys()))
    if not common_keys:
        print("No common tensor keys matched between the two files.")
        return

    global_sum = 0.0
    global_abs_sum = 0.0
    global_numel = 0

    for key in common_keys:
        ta = tensors_a[key]
        tb = tensors_b[key]

        if ta.shape != tb.shape:
            print(f"[SKIP] {key}: shape mismatch {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue

        diff = ta.to(torch.float32) - tb.to(torch.float32)
        mean_diff = diff.mean().item()
        mean_abs_diff = diff.abs().mean().item()

        print(
            f"{key:80s} shape={tuple(diff.shape)} "
            f"mean_diff={mean_diff:.8e} mean_abs_diff={mean_abs_diff:.8e}"
        )

        global_sum += diff.sum().item()
        global_abs_sum += diff.abs().sum().item()
        global_numel += diff.numel()

    if global_numel == 0:
        print("No comparable tensors found after shape checks.")
        return

    print("\n===== Global Summary =====")
    print(f"global_mean_diff={global_sum / global_numel:.8e}")
    print(f"global_mean_abs_diff={global_abs_sum / global_numel:.8e}")


_compare_filtered_tensors(PT_PATH_A, PT_PATH_B)

'''
模型下载：
hf download worstcoder/rcm-Wan \
  Wan2.2-I2V-A14B-low-rCM4.0-merged.pt \
  --repo-type model \
  --local-dir /share/workspace/aigc/rcm/assets/ \
  --max-workers 8

hf download worstcoder/Wan_datasets \
  --repo-type dataset \
  --local-dir /share/workspace/aigc/rcm/assets/dataset/ \
  --include "Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/**"

'''