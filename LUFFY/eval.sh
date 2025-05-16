ROOT=/fs-computility/ai-shen/songxin/LUFFY
DATA=$ROOT/data/valid.all.parquet

OUTPUT_DIR=./results_new/
mkdir -p $OUTPUT_DIR

# If you want to evaluate other models, you can change the model path and name.
#MODEL_PATH=DeepSeek-R1-Distill-Llama-8B

MODEL_NAME=DeepSeek-R1-Distill-Llama-8B



TEMPLATE=no
# if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
#   TEMPLATE=prime
# elif [ $MODEL_NAME == "simple-rl-zero" ]; then
#   TEMPLATE=qwen
# else
#   TEMPLATE=own
# fi

python eval_scripts/generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log
