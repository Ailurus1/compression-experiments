lm_eval --model vllm \
  --model_args pretrained=$MODEL_PATH,add_bos_token=true,attention_backend="FLEX_ATTENTION",tensor_parallel_size=$TP_SIZE \
  --tasks mmlu \
  --limit 12 \
  --output_path ./$RUN_NAME \
  --batch_size 'auto'