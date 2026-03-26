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
        default="/share/Wan2.1-T2V-14B-Diffusers-rcm/",
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--custom_ckpt",
        default="/share/workspace/aigc/rcm/14B_480p_iter_100000_diffusers.safetensors",
        help="Path to custom safetensors checkpoint.",
    )
    return parser.parse_args()


args = parse_args()
model_dir = args.model_dir
transformer_dir = os.path.join(model_dir, "transformer")
custom_ckpt = args.custom_ckpt

if not os.path.exists(transformer_dir):
    raise FileNotFoundError(f"transformer_dir not found: {transformer_dir}")

if not os.path.exists(custom_ckpt):
    raise FileNotFoundError(f"custom_ckpt not found: {custom_ckpt}")

# 找 index.json
index_files = [
    f for f in os.listdir(transformer_dir)
    if f.endswith(".index.json")
]

if len(index_files) != 1:
    raise RuntimeError(
        f"Expected exactly 1 index.json in {transformer_dir}, "
        f"but found {len(index_files)}: {index_files}"
    )

index_file = os.path.join(transformer_dir, index_files[0])
print(f"Using index file: {index_file}")

with open(index_file, "r", encoding="utf-8") as f:
    index_data = json.load(f)

if "weight_map" not in index_data:
    raise KeyError(f"'weight_map' not found in index file: {index_file}")

weight_map = index_data["weight_map"]  # key -> shard file
print(f"Indexed parameters: {len(weight_map)}")

# 1. 加载原始 transformer
print(f"Loading transformer from: {transformer_dir}")
transformer = WanTransformer3DModel.from_pretrained(
    transformer_dir,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True, # 用更省内存的方式加载模型权重（尤其是大模型）
)

orig_sd = transformer.state_dict()
print(f"Original transformer params: {len(orig_sd)}")

# 2. 读取自定义 safetensors
print(f"Loading custom weights from: {custom_ckpt}")
custom_sd = load_file(custom_ckpt)
print(f"Custom ckpt params: {len(custom_sd)}")

# 3. 匹配并替换
matched = []
shape_mismatch = []
not_found = []

for k, v in custom_sd.items():
    if k in orig_sd:
        if tuple(orig_sd[k].shape) == tuple(v.shape):
            orig_sd[k] = v.to(dtype=orig_sd[k].dtype)
            matched.append(k)
        else:
            shape_mismatch.append((k, tuple(orig_sd[k].shape), tuple(v.shape)))
    else:
        not_found.append(k)

print(f"Matched keys: {len(matched)}")
print(f"Shape mismatch keys: {len(shape_mismatch)}")    # 原权重和新权重都有，但形状不匹配
print(f"Not found keys: {len(not_found)}")  # 新权重独有的权重名
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

# 4. 校验 index 中的 key 是否都能在 state_dict 里找到
missing_in_state_dict = [k for k in weight_map if k not in orig_sd]
if missing_in_state_dict:
    print(f"Keys in index but missing in state_dict: {len(missing_in_state_dict)}")
    for k in missing_in_state_dict[:20]:
        print("  ", k)
    raise ValueError("State dict does not fully cover keys from index.json")

# 5. 按原 index 的 weight_map 重新组装各 shard
shard_to_tensors = defaultdict(dict)
for k, shard_name in weight_map.items():
    shard_to_tensors[shard_name][k] = orig_sd[k].contiguous()

print(f"Total shards to overwrite: {len(shard_to_tensors)}")
for shard_name, shard_sd in shard_to_tensors.items():
    print(f"  {shard_name}: {len(shard_sd)} tensors")

# 6. 原地覆盖每个 shard 文件
for shard_name, shard_sd in shard_to_tensors.items():
    shard_path = os.path.join(transformer_dir, shard_name)
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard file not found: {shard_path}")
    print(f"Overwriting shard: {shard_path}")
    save_file(shard_sd, shard_path)

print("Done.")
print("Only sharded safetensors files were overwritten.")
print("Index and config files were not modified.")

"""
python scripts/replace_diffusers.py \
    --model_dir /share/Wan2.1-T2V-14B-Diffusers-rcm/ \
    --custom_ckpt /share/workspace/aigc/rcm/14B_480p_iter_100000_diffusers.safetensors
"""
