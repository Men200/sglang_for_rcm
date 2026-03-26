# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcp_checkpoint_dir", type=str, default="/rcm/outputs/rcm/rCM_Wan/wan2pt1_1pt3B_res480p_t2v_rCM/checkpoints/iter_000005000/model/")
    parser.add_argument("--save_path", type=str, default="/rcm/ckpt/1pt3_480p.safetensors")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32", "keep"],
        help="Output floating tensor dtype.",
    )
    parser.add_argument(
        "--input_prefix",
        type=str,
        default="net_ema.",
        help="Only export keys that start with this prefix.",
    )
    parser.add_argument(
        "--reshape_patch_embed",
        action="store_true",
        help="Reshape patch embedding weight from linear format to convolution format.",
    )
    parser.add_argument(
        "--patch_embed_key",
        type=str,
        default="patch_embedding.weight",
        help="Target key (or suffix) for patch embedding weight.",
    )
    parser.add_argument(
        "--patch_embed_split_dims",
        type=str,
        default="16,1,2,2",
        help="Comma-separated dims used to split linear in_features. Example: 16,1,2,2",
    )
    return parser.parse_args()


def parse_split_dims(split_dims_str: str) -> tuple[int, ...]:
    dims = tuple(int(v.strip()) for v in split_dims_str.split(",") if v.strip())
    if len(dims) == 0:
        raise ValueError("--patch_embed_split_dims must contain at least one integer.")
    if any(d <= 0 for d in dims):
        raise ValueError(f"All split dims must be positive, got: {dims}")
    return dims


def maybe_reshape_patch_embed_weight(
    tensor: torch.Tensor,
    key: str,
    patch_embed_key: str,
    split_dims: tuple[int, ...],
) -> tuple[torch.Tensor, bool]:
    if not (key == patch_embed_key or key.endswith(f".{patch_embed_key}")):
        return tensor, False

    if tensor.ndim != 2:
        raise ValueError(
            f"Patch embedding weight must be 2D before reshape, but got shape {tuple(tensor.shape)} for key: {key}"
        )

    out_dim, flat_in_dim = tensor.shape
    expected_in_dim = math.prod(split_dims)
    if flat_in_dim != expected_in_dim:
        raise ValueError(
            f"Cannot reshape key {key}: linear in_features={flat_in_dim}, "
            f"but product(--patch_embed_split_dims)={expected_in_dim} from {split_dims}"
        )

    reshaped = tensor.reshape(out_dim, *split_dims).contiguous()
    return reshaped, True


def cast_tensor_dtype(tensor: torch.Tensor, dtype_name: str) -> torch.Tensor:
    if not tensor.is_floating_point() or dtype_name == "keep":
        return tensor
    if dtype_name == "bf16":
        return tensor.to(torch.bfloat16)
    if dtype_name == "fp16":
        return tensor.to(torch.float16)
    if dtype_name == "fp32":
        return tensor.to(torch.float32)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


if __name__ == "__main__":
    args = parse_arguments()
    split_dims = parse_split_dims(args.patch_embed_split_dims)

    storage_reader = FileSystemReader(args.dcp_checkpoint_dir)
    state_dict = {}
    _load_state_dict(state_dict, storage_reader=storage_reader, planner=_EmptyStateDictLoadPlanner(), no_dist=True)

    exported_state_dict = {}
    patch_embed_reshaped = False
    for key, value in state_dict.items():
        if key.startswith(args.input_prefix):
            output_key = key[len(args.input_prefix) :]
            output_value = value
            if args.reshape_patch_embed:
                output_value, reshaped = maybe_reshape_patch_embed_weight(
                    tensor=output_value,
                    key=output_key,
                    patch_embed_key=args.patch_embed_key,
                    split_dims=split_dims,
                )
                patch_embed_reshaped = patch_embed_reshaped or reshaped

            exported_state_dict[output_key] = cast_tensor_dtype(output_value, args.dtype)

    if args.reshape_patch_embed and not patch_embed_reshaped:
        raise ValueError(
            f"--reshape_patch_embed is enabled, but key '{args.patch_embed_key}' was not found in exported tensors. "
            "Please check --input_prefix and --patch_embed_key."
        )

    if len(exported_state_dict) == 0:
        raise ValueError(
            f"No tensors were exported. Please check --input_prefix (current: {args.input_prefix}) "
            "and whether the checkpoint contains matching keys."
        )

    save_file(exported_state_dict, args.save_path)
    print(f"Saved {len(exported_state_dict)} tensors to {args.save_path}")
    if args.reshape_patch_embed:
        print(
            f"Reshaped patch embedding key '{args.patch_embed_key}' using split dims {split_dims}."
        )

"""
python scripts/dcp_to_safetensors.py \
    --dcp_checkpoint_dir /share/workspace/aigc/rcm/14B_480p_iter_100000/model/  \
    --save_path /share/workspace/aigc/rcm/14B_480p_iter_100000.safetensors \
    --input_prefix net_ema. \
    --reshape_patch_embed \
    --patch_embed_key patch_embedding.weight \  # 调整 patch 嵌入层权重的形状
    --patch_embed_split_dims 16,1,2,2
"""