export PYTHONPATH="."

#export MODEL_PATH="/share/Wan2.2-I2V-A14B-Diffusers"
#export SGLANG_PERF_LOG_DIR=./sglang_logs
#export MODEL_NAME_LOWER="wan2.2-i2v-a14b"
#export MODEL_SERVICE_NAME="wan2.2-i2v-a14b"

#export MODEL_PATH="/share/Wan2.2-T2V-A14B-Diffusers"
#export SGLANG_PERF_LOG_DIR=./sglang_logs
#export MODEL_NAME_LOWER="wan2.2-t2v-a14b"
#export MODEL_SERVICE_NAME="wan2.2-t2v-a14b"

#export MODEL_PATH="/share/Qwen-Image-Edit-2509"
#export SGLANG_PERF_LOG_DIR=./sglang_logs
#export MODEL_NAME_LOWER="qwen-image-edit-2509"
#export MODEL_SERVICE_NAME="qwen-image-edit-2509"

# export MODEL_PATH="/share/z-image-turbo/"
# export SGLANG_PERF_LOG_DIR=./sglang_logs
# export MODEL_NAME_LOWER="z-image-turbo"
# export MODEL_SERVICE_NAME="z-image-turbo"

#export MODEL_PATH="/share/Qwen-Image/"
#export SGLANG_PERF_LOG_DIR=./sglang_logs
#export MODEL_NAME_LOWER="qwen-image"
#export MODEL_SERVICE_NAME="qwen-image"

export MODEL_PATH="/share/Wan2.2-I2V-A14B-Diffusers-rcm"
export SGLANG_PERF_LOG_DIR=./sglang_logs
export MODEL_NAME_LOWER="wan2.2-i2v-a14b"
export MODEL_SERVICE_NAME="rcm-test"
export USE_SGLANG=true
export SGLANG_NOT_USE_DEFAULT_CACHE_DIT=1 # 随便什么值，设置了就行


export KWS_SERVICE_STAGE=CANDIDATE
export USE_HTTP_MOCK=false
export CUDA_VISIBLE_DEVICES=7
uv run diffusionSrc/server_sglang.py