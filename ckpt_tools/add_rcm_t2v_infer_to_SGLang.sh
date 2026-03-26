# 拉取 diffusion-worker 仓库后请切换到 develop-wxw 分支，下列修改全针对其中选择的 SGLang 版本
git checkout develop-wxw
git clone https://github.com/sgl-project/sglang.git
bash diffusionBin/install_deps.sh
# 其中有指定 SGLang 仓库的指令：git checkout 1d942e4eef5e4ee5ba18f1b8fd134780c5465227

cd sglang/
# 请在 python/sglang/multimodal_gen/runtime/models/schedulers/ 目录下添加调度器文件 scheduling_rcm.py
# 修改 python/sglang/multimodal_gen/runtime/pipelines/wan_pipeline.py 中选择调度器的代码
# 修改 python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py 中调用调度器的代码
# 修改 python/sglang/multimodal_gen/runtime/pipelines_core/stages/latent_preparation.py 中初始化噪声的精度（t2v float32; i2v float64)
# 修改 python/sglang/multimodal_gen/configs/sample/sampling_params.py 中参数覆盖的代码
# 删除 python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py 中覆盖 fps 的代码
# 修改 python/sglang/multimodal_gen/registry.py 中模型匹配的代码（暂时可忽略，但在通过 SGLang 调用 Wan 系列模型时会设置匹配错误）

cd ..

# 调用脚本 dcp_to_safetensors.py ，从蒸馏生成的 DCP 文件中抽取出学生权重，转换为 safetensors 格式
python ckpt_tools/dcp_to_safetensors.py \
    --dcp_checkpoint_dir /share/workspace/aigc/rcm/14B_480p_iter_100000/model/  \
    --save_path /share/workspace/aigc/rcm/14B_480p_iter_100000.safetensors \
    --input_prefix net_ema. \
    --reshape_patch_embed \
    --patch_embed_key patch_embedding.weight \  # 调整 patch 嵌入层权重的形状
    --patch_embed_split_dims 16,1,2,2
# 调用脚本 convert_wan_to_diffusers.py 将权重名改为 diffusers 格式
python ckpt_tools/convert_wan_to_diffusers.py \
    --model_type Wan-T2V-14B \
    --source_path /share/workspace/aigc/rcm/14B_480p_iter_100000.safetensors \
    --output_path /share/workspace/aigc/rcm/14B_480p_iter_100000_diffusers.safetensors

# 同时请准备好 Wan2.1-T2V-14B-Diffusers 模型权重
# 从 https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/tree/main 下载，或者直接在/share目录下复制
cp -r /share/Wan2.1-T2V-14B-Diffusers /share/Wan2.1-T2V-14B-Diffusers-rcm
# 调用脚本 replace_diffusers.py 替换 transformer 目录下的 safetensors 文件中的同名权重【请保证环境中已安装了 diffusers 库】
# 拿新权重一个个去 transformer 权重中找，同名但形状不匹配的会自动跳过
uv run python ckpt_tools/replace_diffusers.py \
    --model_dir /share/Wan2.1-T2V-14B-Diffusers-rcm/ \
    --custom_ckpt /share/workspace/aigc/rcm/14B_480p_iter_100000_diffusers.safetensors

# 调用指令如下
export PYTHONPATH="."
export MODEL_PATH="/share/Wan2.1-T2V-14B-Diffusers-rcm"
export SGLANG_PERF_LOG_DIR=./sglang_logs
export MODEL_NAME_LOWER="wan2.1-t2v-14b"
export MODEL_SERVICE_NAME="rcm-test"
export USE_SGLANG=true
export SGLANG_NOT_USE_DEFAULT_CACHE_DIT=1 # 随便什么值，设置了就行
export KWS_SERVICE_STAGE=CANDIDATE
export USE_HTTP_MOCK=false
export CUDA_VISIBLE_DEVICES=7
uv run diffusionSrc/server_sglang.py

# url 请求指令
curl --location --request POST 'http://wanqing-aigc-workflow.test.gifshow.com/openapi/v1/text2video/generations' \
--header 'X-Wanqing-User: qianyaotian' \
--header 'X-Wanqing-Project: test-project' \
--header 'x-api-key: 7fd5e3f2-c17f-451b-b29e-df84a7f50197' \
--header 'x-ks-provider: internal' \
--header 'Content-Type: application/json' \
--data-raw '{
  "prompt": "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
  "model": "wan2.1-t2v-14b",
  "negative_prompt": "",
  "guidance_scale": 1.0,
  "size": "832*480",
  "duration": 5,
  "fps": 16,
  "model_service": "rcm-test",
  "num_inference_steps": 4,
  "seed": 1
}'

# 针对 I2V ===================================
cd sglang/
# 修改 python/sglang/multimodal_gen/runtime/pipelines/wan_i2v_pipeline.py 中选择调度器的代码

cd ..

# 同样下载或复制一个 /share/Wan2.2-I2V-A14B-Diffusers-rcm
cp -r /share/Wan2.2-I2V-A14B-Diffusers /share/Wan2.2-I2V-A14B-Diffusers-rcm
# 假设已像前文那样准备好 diffusers 格式的自定义权重文件 14B_480p_iter_100000_diffusers.safetensors
# 调用脚本 replace_diffusers_i2v.py 对权重增量进行加权，其中w_high=4.0，w_low=1.0
# [Wan2.2 I2V rCM w] = [Wan2.2 I2V base] + w * ([Wan2.1 T2V rCM] - [Wan2.1 T2V base])
uv run python ckpt_tools/replace_diffusers_i2v.py \
    --model_dir /share/Wan2.2-I2V-A14B-Diffusers-rcm/ \
    --model_dir_t2v /share/Wan2.1-T2V-14B-Diffusers/ \
    --custom_ckpt /share/workspace/aigc/rcm/14B_480p_iter_100000_diffusers.safetensors
# 会自动忽略形状不匹配的 ('patch_embedding.weight', (5120, 36, 1, 2, 2), (5120, 16, 1, 2, 2))

# 调用指令和 t2v 基本一样，修改 MODEL_PATH 和 MODEL_NAME_LOWER 即可
export MODEL_PATH="/share/Wan2.2-I2V-A14B-Diffusers-rcm"
export MODEL_NAME_LOWER="wan2.2-i2v-a14b"

# url 请求指令
curl --location --request POST 'http://wanqing-aigc-workflow.test.gifshow.com/openapi/v1/image2video/generations' \
--header 'X-Wanqing-User: qianyaotian' \
--header 'X-Wanqing-Project: test-project' \
--header 'x-api-key: 7fd5e3f2-c17f-451b-b29e-df84a7f50197' \
--header 'x-ks-provider: internal' \
--header 'Content-Type: application/json' \
--data-raw '{
  "prompt": "POV selfie video, ultra-messy and extremely fast. A white cat in sunglasses stands on a surfboard with a neutral look when the board suddenly whips sideways, throwing cat and camera into the water; the frame dives sharply downward, swallowed by violent bursts of bubbles, spinning turbulence, and smeared water streaks as the camera sinks. Shadows thicken, pressure ripples distort the edges, and loose bubbles rush upward past the lens, showing the camera is still sinking. Then the cat kicks upward with explosive speed, dragging the view through churning bubbles and rapidly brightening water as sunlight floods back in; the camera races upward, water streaming off the lens, and finally breaks the surface in a sudden blast of light and spray, snapping back into a crooked, frantic selfie as the cat resurfaces.",
  "model": "wan2.2-i2v-a14b",
  "negative_prompt": "",
  "guidance_scale": 1.0,
  "size": "720*1280",
  "duration": 5,
  "fps": 16,
  "model_service": "rcm-test",
  "num_inference_steps": 4,
  "seed": 1,
  "first_frame": "https://bs3-hb1.staging.kuaishou.com/infra-wanqing-aigc-workflow/test/images/first_frame.png"
}'

# 返回结果的 url 指令均为
curl --location --request GET 'http://wanqing-aigc-workflow.test.gifshow.com/openapi/v1/task/task-kvn2w5-1774532212969/status' \
--header 'X-Wanqing-User: qianyaotian' \
--header 'X-Wanqing-Project: testProject' \
--header 'x-api-key: 7fd5e3f2-c17f-451b-b29e-df84a7f50197' \
--header 'x-ks-provider: internal'