lm_eval --model $BACKEND \
  --model_args pretrained=$MODEL_PATH,add_bos_token=true,$MODEL_PARAMS \ 
  --tasks mmlu \
  --limit 12 \
  --output_path ./$RUN_NAME \
  --batch_size 'auto'
