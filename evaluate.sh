lm_eval --model vllm \
  --model_args pretrained="/kaggle/working/Qwen3-8B-GPTQ-W4A16",add_bos_token=true,attention_backend="FLEX_ATTENTION",tensor_parallel_size=2 \
  --tasks mmlu \
  --limit 12 \
  --batch_size 'auto'