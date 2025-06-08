export HF_HOME= #enter your own path to the hf home
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --dtype auto \
  --api-key token-abc123 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --gpu-memory-utilization 0.75 \
  --max-model-len 15360 \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 15360 \
  --max-num-seqs 16
