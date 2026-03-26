import argparse
import pathlib
from typing import Any, Dict, Tuple
import torch

from safetensors.torch import load_file, save_file


# ========== 键名重命名映射 ==========
TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
}

VACE_TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
    "before_proj": "proj_in",
    "after_proj": "proj_out",
}

ANIMATE_TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "cross_attn.k_img": "attn2.to_k_img",
    "cross_attn.v_img": "attn2.to_v_img",
    "cross_attn.norm_k_img": "attn2.norm_k_img",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
    "motion_encoder.enc.fc": "motion_encoder.motion_network",
    "motion_encoder.dec.direction.weight": "motion_encoder.motion_synthesis_weight",
    "face_encoder.conv1_local.conv": "face_encoder.conv1_local",
    "face_encoder.conv2.conv": "face_encoder.conv2",
    "face_encoder.conv3.conv": "face_encoder.conv3",
}


def convert_animate_motion_encoder_weights(key: str, state_dict: Dict[str, Any], final_conv_idx: int = 8) -> None:
    """Convert motion encoder weights for Animate model."""
    if ".weight" not in key and ".bias" not in key and ".kernel" not in key:
        return

    if ".kernel" in key and "motion_encoder" in key:
        state_dict.pop(key, None)
        return

    if ".enc.net_app.convs." in key and (".weight" in key or ".bias" in key):
        parts = key.split(".")
        convs_idx = parts.index("convs") if "convs" in parts else -1
        if convs_idx >= 0 and len(parts) - convs_idx >= 2:
            bias = False
            sequential_idx = int(parts[convs_idx + 1])
            if sequential_idx == 0:
                if key.endswith(".weight"):
                    new_key = "motion_encoder.conv_in.weight"
                elif key.endswith(".bias"):
                    new_key = "motion_encoder.conv_in.act_fn.bias"
                    bias = True
            elif sequential_idx == final_conv_idx:
                if key.endswith(".weight"):
                    new_key = "motion_encoder.conv_out.weight"
            else:
                prefix = "motion_encoder.res_blocks."
                layer_name = parts[convs_idx + 2]
                if layer_name == "skip":
                    layer_name = "conv_skip"
                if key.endswith(".weight"):
                    param_name = "weight"
                elif key.endswith(".bias"):
                    param_name = "act_fn.bias"
                    bias = True
                suffix_parts = [str(sequential_idx - 1), layer_name, param_name]
                suffix = ".".join(suffix_parts)
                new_key = prefix + suffix

            param = state_dict.pop(key)
            if bias:
                param = param.squeeze()
            state_dict[new_key] = param
            return
    return


def convert_animate_face_adapter_weights(key: str, state_dict: Dict[str, Any]) -> None:
    """Convert face adapter weights for the Animate model."""
    if ".weight" not in key and ".bias" not in key:
        return

    prefix = "face_adapter."
    if ".fuser_blocks." in key:
        parts = key.split(".")
        module_list_idx = parts.index("fuser_blocks") if "fuser_blocks" in parts else -1
        if module_list_idx >= 0 and (len(parts) - 1) - module_list_idx == 3:
            block_idx = parts[module_list_idx + 1]
            layer_name = parts[module_list_idx + 2]
            param_name = parts[module_list_idx + 3]

            if layer_name == "linear1_kv":
                layer_name_k = "to_k"
                layer_name_v = "to_v"
                suffix_k = ".".join([block_idx, layer_name_k, param_name])
                suffix_v = ".".join([block_idx, layer_name_v, param_name])
                new_key_k = prefix + suffix_k
                new_key_v = prefix + suffix_v

                kv_proj = state_dict.pop(key)
                k_proj, v_proj = torch.chunk(kv_proj, 2, dim=0)
                state_dict[new_key_k] = k_proj
                state_dict[new_key_v] = v_proj
                return
            else:
                if layer_name == "q_norm":
                    new_layer_name = "norm_q"
                elif layer_name == "k_norm":
                    new_layer_name = "norm_k"
                elif layer_name == "linear1_q":
                    new_layer_name = "to_q"
                elif layer_name == "linear2":
                    new_layer_name = "to_out"

                suffix_parts = [block_idx, new_layer_name, param_name]
                suffix = ".".join(suffix_parts)
                new_key = prefix + suffix
                state_dict[new_key] = state_dict.pop(key)
                return
    return


TRANSFORMER_SPECIAL_KEYS_REMAP = {}
VACE_TRANSFORMER_SPECIAL_KEYS_REMAP = {}
ANIMATE_TRANSFORMER_SPECIAL_KEYS_REMAP = {
    "motion_encoder": convert_animate_motion_encoder_weights,
    "face_adapter": convert_animate_face_adapter_weights,
}


def update_state_dict_(state_dict: Dict[str, Any], old_key: str, new_key: str) -> dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def get_rename_dicts(model_type: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Get rename dicts for different model types."""
    if "Animate" in model_type:
        return ANIMATE_TRANSFORMER_KEYS_RENAME_DICT, ANIMATE_TRANSFORMER_SPECIAL_KEYS_REMAP
    elif "VACE" in model_type:
        return VACE_TRANSFORMER_KEYS_RENAME_DICT, VACE_TRANSFORMER_SPECIAL_KEYS_REMAP
    else:
        return TRANSFORMER_KEYS_RENAME_DICT, TRANSFORMER_SPECIAL_KEYS_REMAP


def convert_state_dict_keys(source_path: str, model_type: str) -> Dict[str, torch.Tensor]:
    """
    将本地权重文件转换为 diffusers 格式的键名
    """
    source_path = pathlib.Path(source_path)

    if not source_path.exists():
        raise FileNotFoundError(f"File not found: {source_path}")

    # 获取重命名规则
    RENAME_DICT, SPECIAL_KEYS_REMAP = get_rename_dicts(model_type)

    # 加载原始权重
    print(f"Loading weights from: {source_path}")
    original_state_dict = load_file(source_path)
    print(f"Loaded {len(original_state_dict)} tensors")

    # 键名重命名
    print("Renaming keys...")
    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_(original_state_dict, key, new_key)

    # 特殊键的处理（如 motion_encoder, face_adapter）
    print("Applying special key remapping...")
    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    print(f"Conversion complete! Total tensors: {len(original_state_dict)}")
    return original_state_dict


def convert_and_save(source_path: str, output_path: str, model_type: str):
    """转换并保存权重"""
    # 转换键名
    state_dict = convert_state_dict_keys(source_path, model_type)

    # 保存为 safetensors
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {output_path}")
    save_file(state_dict, str(output_path))
    print("Done!")


def get_args():
    parser = argparse.ArgumentParser(description="Convert local Wan checkpoint to diffusers format")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Model type: Wan-T2V-1.3B, Wan-T2V-14B, Wan-I2V-14B-480p, etc.")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to the source safetensors file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the converted safetensors file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    convert_and_save(args.source_path, args.output_path, args.model_type)

"""
python scripts/convert_to_diffusers.py \
    --model_type Wan-T2V-14B \
    --source_path /share/workspace/aigc/rcm/14B_480p_iter_100000.safetensors \
    --output_path /share/workspace/aigc/rcm/14B_480p_iter_100000_diffusers.safetensors
"""