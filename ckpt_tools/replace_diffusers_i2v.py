import os
import json
import argparse
import torch
from collections import defaultdict
from safetensors.torch import load_file, save_file
from diffusers import WanTransformer3DModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replace diffusers transformer shard weights with a custom checkpoint."
    )
    parser.add_argument(
        "--model_dir",
        default="/share/Wan2.2-I2V-A14B-Diffusers-rcm",
        help="Path to the base model directory.",
    )
    parser.add_argument(
        "--model_dir_t2v",
        default="/share/Wan2.1-T2V-14B-Diffusers",
        help="Path to the T2V model directory.",
    )
    parser.add_argument(
        "--custom_ckpt",
        default="/share/workspace/aigc/rcm/14B_480p_iter_100000_diffusers.safetensors",
        help="Path to custom safetensors checkpoint.",
    )
    return parser.parse_args()


args = parse_args()
model_dir = args.model_dir
model_dir_t2v = args.model_dir_t2v
transformer_dir_hig = os.path.join(model_dir, "transformer")
transformer_dir_low = os.path.join(model_dir, "transformer_2")
transformer_dir_t2v = os.path.join(model_dir_t2v, "transformer")
custom_ckpt = args.custom_ckpt

if not os.path.exists(transformer_dir_hig):
    raise FileNotFoundError(f"transformer_dir_hig not found: {transformer_dir_hig}")

if not os.path.exists(transformer_dir_low):
    raise FileNotFoundError(f"transformer_dir_low not found: {transformer_dir_low}")

if not os.path.exists(transformer_dir_t2v):
    raise FileNotFoundError(f"transformer_dir_t2v not found: {transformer_dir_t2v}")
    
if not os.path.exists(custom_ckpt):
    raise FileNotFoundError(f"custom_ckpt not found: {custom_ckpt}")


# [Wan2.2 I2V rCM w] = [Wan2.2 I2V base] + w * ([Wan2.1 T2V rCM] - [Wan2.1 T2V base])
def merge_ckpt_delta(base_dir, ckpt_delta, w):
    # 找 index.json 用来分块存储
    base_index_files = [
        f for f in os.listdir(base_dir)
        if f.endswith(".index.json")
    ]

    if len(base_index_files) != 1:
        raise RuntimeError(
            f"Expected exactly 1 index.json in {base_dir}, "
            f"but found {len(base_index_files)}: {base_index_files}"
        )
    
    base_index_file = os.path.join(base_dir, base_index_files[0])
    print(f"Using index file: {base_index_file}")

    with open(base_index_file, "r", encoding="utf-8") as f:
        base_index_data = json.load(f)

    if "weight_map" not in base_index_data:
        raise KeyError(f"'weight_map' not found in index file: {base_index_file}")

    weight_map = base_index_data["weight_map"]  # key -> shard file
    print(f"Indexed parameters: {len(weight_map)}")

    # 1. 加载 base transformer
    print(f"Loading transformer from: {base_dir}")
    base_model = WanTransformer3DModel.from_pretrained(
        base_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    base_sd = base_model.state_dict()
    print(f"Original transformer params: {len(base_sd)}")

    # 2. 加权合并权重增量
    matched = []
    shape_mismatch = []
    not_found = []

    for k, v in ckpt_delta.items():
        if k in base_sd:
            if tuple(base_sd[k].shape) == tuple(v.shape):
                base_sd[k] += (w * v).to(dtype=base_sd[k].dtype)
                matched.append(k)
            else:
                shape_mismatch.append((k, tuple(base_sd[k].shape), tuple(v.shape)))
        else:
            not_found.append(k)
    
    print(f"Matched keys: {len(matched)}")
    print(f"Shape mismatch keys: {len(shape_mismatch)}")
    print(f"Not found keys: {len(not_found)}")
    print(f"Matched ratio: {len(matched)}/{len(base_sd)} = {len(matched)/len(base_sd):.2%}")

    if matched:
        print("First 20 matched keys:")
        for k in matched[:20]:
            print("  ", k)

    if shape_mismatch:
        print("First 20 shape mismatch keys:")
        for item in shape_mismatch[:20]:
            print("  ", item)

    if not_found:
        print("First 20 not found keys:")
        for k in not_found[:20]:
            print("  ", k)

    if len(matched) == 0:
        raise ValueError("No matched keys found. Abort overwriting sharded weights.")
    
    # 3. 校验 index 中的 key 是否都能在 state_dict 里找到
    missing_in_state_dict = [k for k in weight_map if k not in base_sd]
    if missing_in_state_dict:
        print(f"Keys in index but missing in i2v's state_dict: {len(missing_in_state_dict)}")
        for k in missing_in_state_dict[:20]:
            print("  ", k)
        raise ValueError("State dict does not fully cover keys from index.json")

    # 4. 按原 index 的 weight_map 重新组装各 shard
    shard_to_tensors = defaultdict(dict)
    for k, shard_name in weight_map.items():
        shard_to_tensors[shard_name][k] = base_sd[k].contiguous()

    print(f"Total shards to overwrite: {len(shard_to_tensors)}")
    for shard_name, shard_sd in shard_to_tensors.items():
        print(f"  {shard_name}: {len(shard_sd)} tensors")

    # 5. 原地覆盖每个 shard 文件
    for shard_name, shard_sd in shard_to_tensors.items():
        shard_path = os.path.join(base_dir, shard_name)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
        print(f"Overwriting shard: {shard_path}")
        save_file(shard_sd, shard_path)


print(f"Computing delta weights: {transformer_dir_t2v} -> {custom_ckpt} ======================================")

# 1. 加载t2v transformer
print(f"Loading transformer from: {transformer_dir_t2v}")
transformer = WanTransformer3DModel.from_pretrained(
    transformer_dir_t2v,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

orig_sd = transformer.state_dict()
print(f"Original transformer params: {len(orig_sd)}")

# 2. 读取自定义 safetensors
print(f"Loading custom weights from: {custom_ckpt}")
custom_sd = load_file(custom_ckpt)
print(f"Custom ckpt params: {len(custom_sd)}")

# 3. 匹配并计算 Delta
matched = []
shape_mismatch = []
not_found = []

for k, v in custom_sd.items():
    if k in orig_sd:
        if tuple(orig_sd[k].shape) == tuple(v.shape):
            # Save delta tensor for matched keys: delta = custom - t2v.
            orig_sd[k] = v.to(dtype=orig_sd[k].dtype) - orig_sd[k]
            matched.append(k)
        else:
            shape_mismatch.append((k, tuple(orig_sd[k].shape), tuple(v.shape)))
    else:
        not_found.append(k)

print(f"Matched keys: {len(matched)}")
print(f"Shape mismatch keys: {len(shape_mismatch)}")
print(f"Not found keys: {len(not_found)}")
print(f"Matched ratio: {len(matched)}/{len(orig_sd)} = {len(matched)/len(orig_sd):.2%}")

if matched:
    print("First 20 matched keys:")
    for k in matched[:20]:
        print("  ", k)

if shape_mismatch:
    print("First 20 shape mismatch keys:")
    for item in shape_mismatch[:20]:
        print("  ", item)

if not_found:
    print("First 20 not found keys:")
    for k in not_found[:20]:
        print("  ", k)

if len(matched) == 0:
    raise ValueError("No matched keys found. Abort overwriting sharded weights.")

print(f"Computing merged weights for high noise model with w=4.0 ======================================")
merge_ckpt_delta(base_dir=transformer_dir_hig, ckpt_delta=orig_sd, w=4.0)

print(f"Computing merged weights for low noise model with w=1.0 ======================================")
merge_ckpt_delta(base_dir=transformer_dir_low, ckpt_delta=orig_sd, w=1.0)


print("Done.")
print("Only sharded safetensors files were overwritten.")
print("Index and config files were not modified.")

"""
uv run python ckpt_tools/replace_diffusers_i2v.py \
    --model_dir /share/Wan2.2-I2V-A14B-Diffusers-rcm/ \
    --model_dir_t2v /share/Wan2.1-T2V-14B-Diffusers/ \
    --custom_ckpt /share/workspace/aigc/rcm/14B_480p_iter_100000_diffusers.safetensors
"""
