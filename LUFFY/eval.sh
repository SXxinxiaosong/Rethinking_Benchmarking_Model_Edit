ROOT=/fs-computility/ai-shen/songxin/LUFFY
DATA=$ROOT/data/valid.all.parquet

OUTPUT_DIR=./results_new/
mkdir -p $OUTPUT_DIR

# If you want to evaluate other models, you can change the model path and name.
#/fs-computility/ai-shen/songxin/EasyEdit_for_reasoning_LLM/examples/saved_model/DeepSeek-R1-Distill-Llama-8B_ROME_2025-05-13_09-54-48_step10
#MODEL_PATH=/fs-computility/ai-shen/songxin/models/DeepSeek-R1-Distill-Llama-8B
#MODEL_PATH=/fs-computility/ai-shen/songxin/EasyEdit_for_reasoning_LLM/examples/saved_model/DeepSeek-R1-Distill-Llama-8B_ROME_2025-05-13_09-54-48_step10
#MODEL_PATH=/fs-computility/ai-shen/songxin/EasyEdit_for_reasoning_LLM/examples/saved_model/DeepSeek-R1-Distill-Llama-8B_ROME_2025-05-13_10-24-33_step100
#MODEL_PATH=/fs-computility/ai-shen/songxin/EasyEdit_for_reasoning_LLM/examples/saved_model/DeepSeek-R1-Distill-Llama-8B_AlphaEdit_2025-05-13_09-57-06_step10
#MODEL_PATH=/fs-computility/ai-shen/songxin/EasyEdit_for_reasoning_LLM/examples/saved_model/DeepSeek-R1-Distill-Llama-8B_MEND_2025-05-13_09-57-21_step10
#MODEL_PATH=/fs-computility/ai-shen/songxin/EasyEdit_for_reasoning_LLM/examples/saved_model/DeepSeek-R1-Distill-Llama-8B_RECT_2025-05-13_10-04-28_step10

#MODEL_PATH=/fs-computility/ai-shen/songxin/EasyEdit_for_reasoning_LLM/examples/saved_model/DeepSeek-R1-Distill-Llama-8B_AlphaEdit_2025-05-13_10-25-33_step100
MODEL_PATH=/fs-computility/ai-shen/songxin/EasyEdit_for_reasoning_LLM/examples/saved_model/DeepSeek-R1-Distill-Llama-8B_MEND_2025-05-13_09-57-22_step100
#MODEL_PATH=/fs-computility/ai-shen/songxin/EasyEdit_for_reasoning_LLM/examples/saved_model/DeepSeek-R1-Distill-Llama-8B_RECT_2025-05-13_10-30-48_step100
MODEL_NAME=DeepSeek-R1-Distill-Llama-8B_MEND_100



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