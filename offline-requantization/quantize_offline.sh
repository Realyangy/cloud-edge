#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${bundle_dir}/.." && pwd)"
model_dir="${project_dir}/model"
output_dir="${1:-${bundle_dir}/output}"
jobs="${AIOPS_BUILD_JOBS:-4}"
python_cmd="${PYTHON:-python3}"

if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  printf 'AIOPS_BUILD_JOBS must be a positive integer\n' >&2
  exit 2
fi

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
model_path="${output_dir}/aiops-qwen25-0.5b-Q6_K.gguf"
if [[ -e "$model_path" ]]; then
  printf 'Output already exists: %s\n' "$model_path" >&2
  exit 2
fi

if [[ ! -f "${model_dir}/config.json" || ! -f "${model_dir}/model.safetensors" ]]; then
  printf 'Merged Hugging Face model not found: %s\n' "$model_dir" >&2
  exit 2
fi

work_dir="$(mktemp -d "${output_dir}/.quantize.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT
tar -xzf "${bundle_dir}/source/llama.cpp-091b6a7-source.tar.gz" -C "$work_dir"

f16_model="${work_dir}/aiops-qwen25-0.5b-f16.gguf"
"$python_cmd" "${work_dir}/llama.cpp/convert_hf_to_gguf.py" \
  "$model_dir" \
  --outfile "$f16_model" \
  --outtype f16

cmake -S "${work_dir}/llama.cpp" -B "${work_dir}/build" \
  -DGGML_CUDA=OFF -DGGML_CANN=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${work_dir}/build" --target llama-quantize -j "$jobs"

"${work_dir}/build/bin/llama-quantize" \
  "$f16_model" \
  "$model_path" Q6_K "$jobs"

printf 'Quantized model: %s\n' "$model_path"
