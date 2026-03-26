#!/usr/bin/env bash
set -euo pipefail

# 固定输出（会覆盖）
SAVE_PATH="/rcm/ckpt/1pt3_480p.safetensors"
DIFFUSERS_DIR="/rcm/ckpt/diffusers_ckpt"

# 源 checkpoints 目录（遍历其下所有 iter_*）
CKPT_ROOT="/share/workspace/aigc/rcm/outputs/rcm/rCM_Wan/wan2pt1_1pt3B_res480p_t2v_rCM/checkpoints/"

# 1) 进入 diffusers_ckpt 目录，需保证事先生成 Modelfile，已登录
cd "${DIFFUSERS_DIR}"

# 2) 遍历所有 iter_* 目录（按名称排序）
shopt -s nullglob
for iter_path in "${CKPT_ROOT}"/iter_*; do
  [[ -d "${iter_path}" ]] || continue

  iter_dir="$(basename "${iter_path}")"          # e.g. iter_000025000
  DCP_DIR="${iter_path}/model/"                  # .../iter_xxx/model/
  IMAGE_TAG="registry-model.corp.kuaishou.com/wq-test/1pt3_t2v_480p_${iter_dir}:v0"

  # 如果没有 model 目录就跳过
  if [[ ! -d "${DCP_DIR}" ]]; then
    echo "==== Skip ${iter_dir}: missing ${DCP_DIR} ===="
    continue
  fi

  echo "==== Processing ${iter_dir} ===="
  echo "DCP_DIR=${DCP_DIR}"
  echo "IMAGE_TAG=${IMAGE_TAG}"

  # 1) dcp -> safetensors
  python /rcm/dull_ckpt/dcp_to_safetensors.py \
    --dcp_checkpoint_dir "${DCP_DIR}" \
    --save_path "${SAVE_PATH}" \
    --input_prefix net_ema. \
    --reshape_patch_embed \
    --patch_embed_key patch_embedding.weight \
    --patch_embed_split_dims 16,1,2,2

  # 2) wan -> diffusers
  python /rcm/dull_ckpt/convert_wan_to_diffusers.py \
    --model_type Wan-T2V-1.3B \
    --source_path "${SAVE_PATH}" \
    --output_path "./1pt3_480p_diffusers.safetensors"

  # 3) build：在 diffusers_ckpt 目录下执行，上下文用 .
  modctl build -t "${IMAGE_TAG}" --output-remote .
done