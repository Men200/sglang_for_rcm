#!/usr/bin/env bash
set -euo pipefail

# 固定输出（会覆盖）
PT_DIR="/rcm/ckpt/pt_ckpt"

# 源 checkpoints 目录（遍历其下所有 iter_*）
CKPT_ROOT="/rcm/outputs/rcm/rCM_Wan/wan2pt1_1pt3B_res480p_t2v_rCM/checkpoints"

# 1) 进入 pt_ckpt 目录，需保证事先生成 Modelfile，已登录
cd "${PT_DIR}"

# 2) 遍历所有 iter_* 目录（按名称排序）
shopt -s nullglob
for iter_path in "${CKPT_ROOT}"/iter_*; do
  [[ -d "${iter_path}" ]] || continue

  iter_dir="$(basename "${iter_path}")"                 # iter_000028500
  DCP_DIR="${iter_path}/model"
  IMAGE_TAG="registry-model.corp.kuaishou.com/wq-test/pt_1pt3_t2v_480p_${iter_dir}:v0"

  # 没有 model 目录就跳过
  if [[ ! -d "${DCP_DIR}" ]]; then
    echo "==== Skip ${iter_dir}: missing ${DCP_DIR} ===="
    continue
  fi

  echo "==== Processing ${iter_dir} ===="
  echo "DCP_DIR=${DCP_DIR}"
  echo "IMAGE_TAG=${IMAGE_TAG}"

  # 1) dcp -> pth
  python /rcm/scripts/dcp_to_pth.py \
    --dcp_checkpoint_dir "${DCP_DIR}" \
    --save_path "./1pt3_480p.pt"

  # 2) build：在 pt_ckpt 目录下执行，上下文用 .
  modctl build -t "${IMAGE_TAG}" --output-remote .
done

'''
modctl modelfile generate .		# 当前目录就是放置这个模型的目录
modctl login -u model-admin -p '6&95u6GnpbQn' registry-model.corp.kuaishou.com
modctl build -t registry-model.corp.kuaishou.com/wq-test/pt_14_t2v_480p_iter_000032000:v0 --output-remote .

python /rcm/scripts/dcp_to_pth.py --dcp_checkpoint_dir /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000086000/model/ --save_path ./14_480p.pt
modctl build -t registry-model.corp.kuaishou.com/wq-test/pt_14_t2v_480p_iter_000086000:v0 --output-remote .
rm -rf /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000084000
python /rcm/scripts/dcp_to_pth.py --dcp_checkpoint_dir /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000088000/model/ --save_path ./14_480p.pt
modctl build -t registry-model.corp.kuaishou.com/wq-test/pt_14_t2v_480p_iter_000088000:v0 --output-remote .
rm -rf /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000086000
python /rcm/scripts/dcp_to_pth.py --dcp_checkpoint_dir /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000090000/model/ --save_path ./14_480p.pt
modctl build -t registry-model.corp.kuaishou.com/wq-test/pt_14_t2v_480p_iter_000090000:v0 --output-remote .
rm -rf /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000088000
python /rcm/scripts/dcp_to_pth.py --dcp_checkpoint_dir /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000092000/model/ --save_path ./14_480p.pt
modctl build -t registry-model.corp.kuaishou.com/wq-test/pt_14_t2v_480p_iter_000092000:v0 --output-remote .
rm -rf /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000090000
python /rcm/scripts/dcp_to_pth.py --dcp_checkpoint_dir /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000094000/model/ --save_path ./14_480p.pt
modctl build -t registry-model.corp.kuaishou.com/wq-test/pt_14_t2v_480p_iter_000094000:v0 --output-remote .
rm -rf /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000092000
python /rcm/scripts/dcp_to_pth.py --dcp_checkpoint_dir /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000096000/model/ --save_path ./14_480p.pt
modctl build -t registry-model.corp.kuaishou.com/wq-test/pt_14_t2v_480p_iter_000096000:v0 --output-remote .
rm -rf /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000094000
python /rcm/scripts/dcp_to_pth.py --dcp_checkpoint_dir /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000098000/model/ --save_path ./14_480p.pt
modctl build -t registry-model.corp.kuaishou.com/wq-test/pt_14_t2v_480p_iter_000098000:v0 --output-remote .
rm -rf /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000096000
python /rcm/scripts/dcp_to_pth.py --dcp_checkpoint_dir /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000100000/model/ --save_path ./14_480p.pt
modctl build -t registry-model.corp.kuaishou.com/wq-test/pt_14_t2v_480p_iter_000100000:v0 --output-remote .
rm -rf /rcm/outputs/rcm/rCM_Wan/wan2pt1_14B_res480p_t2v_rCM/checkpoints/iter_000098000
'''