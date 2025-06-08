#!/bin/bash
# Exit on error
set -e
export WANDB_MODE="disabled"
# Define variables
FOLDER_NAME="Experiment"
CUDA_DEVICE=3
ASQA_DATA="../data/asqa.json"
MODEL_PATH="../roberta/roberta-squad"
VAL_FILE="../output_dir/asqa/${FOLDER_NAME}/qa_format/qa.json"
PROCESSED="../output_dir/asqa/${FOLDER_NAME}/processed_asqa.json"
OUTPUT_DIR="../output_dir/asqa/${FOLDER_NAME}"
MAX_SEQ_LEN=5120
NULL_THRESHOLD=0

# Check if files exist
for file in "$ASQA_DATA" "$MODEL_PATH" "$VAL_FILE"; do
    if [ ! -e "$file" ]; then
        echo "Error: File $file does not exist"
        exit 1
    fi
done
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python transformers/examples/pytorch/question-answering/run_qa.py \
    --model_name_or_path ${MODEL_PATH} \
    --validation_file ${VAL_FILE} \
    --do_eval \
    --version_2_with_negative \
    --max_seq_length ${MAX_SEQ_LEN} \
    --output_dir ${OUTPUT_DIR}/disF1 \
    --null_score_diff_threshold ${NULL_THRESHOLD}
# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if the first command succeeded
if [ $? -ne 0 ]; then
    echo "Error: QA evaluation failed"
    exit 1
fi

# Run the script command
python ../dev/src/eval/disambigF1.py --qa-path ${VAL_FILE} --asqa-path ${ASQA_DATA}  --processed-path ${PROCESSED} --predictions-path ${OUTPUT_DIR}/disF1/eval_predictions.json > ${OUTPUT_DIR}/test.txt

# Check if the second command succeeded
if [ $? -ne 0 ]; then
    echo "Error: Script execution failed"
    exit 1
fi

echo "All commands completed successfully"