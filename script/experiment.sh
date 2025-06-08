#!/bin/bash


export URL="$openai_base_url"
export MODEL="$gpt"
export OPENAI_API_KEY="$openai"


# Exit on error
set -e
FOLDER_NAME="experiment"
# Define variables
AMBIG_DATA="../data/AmbigDoc_test.json"
AMBIG_VAL_FILE="../output_dir/AmbigDoc/${FOLDER_NAME}"
DATA="../data/hotpot_dev_fullwiki_v1.json"
WIKI_DATA="../data/wiki_dev.json"
MUSIQUE_DATA="../data/musique_full_v1.0_dev.json"
VAL_FILE="../output_dir/hotpop/${FOLDER_NAME}"
ASQA_DATA="../data/"
VAL_FILE_ASQA="../output_dir/asqa/${FOLDER_NAME}"
MAX_SAMPLE=500

export PYSERINI_CACHE='../corups/'
export KMP_DUPLICATE_LIB_OK='True'
export from_flax='True'
echo "complete asqa"

#
python ../src/asqa_inference.py --input_dir ${ASQA_DATA} --output_dir ${VAL_FILE_ASQA} --max_samples ${MAX_SAMPLE}
echo "starting hotpotqa_inference"
python ../src/hotpotqa_inference.py --input_dir ${DATA} --output_dir ${VAL_FILE}_hotpop --max_samples ${MAX_SAMPLE}  > TreeAct_output.txt
python ../src/eval/eval_multi_hop.py ${VAL_FILE}_hotpop/formated_hotpotqa.json ${DATA} > ${VAL_FILE}_hotpop/result.txt
echo "complete hotpotQA, starting wikiQA"
python ../src/hotpotqa_inference.py --input_dir ${WIKI_DATA} --output_dir ${VAL_FILE}_wiki --max_samples ${MAX_SAMPLE}
python ../src/eval/eval_multi_hop.py ${VAL_FILE}_wiki/formated_hotpotqa.json ${WIKI_DATA} > ${VAL_FILE}_wiki/result.txt
echo "complete wikiQA, starting musique qa"
python ../src/hotpotqa_inference.py --input_dir ${MUSIQUE_DATA} --output_dir ${VAL_FILE}_misque --max_samples ${MAX_SAMPLE} 
python ../src/eval/eval_multi_hop.py ${VAL_FILE}_misque/formated_hotpotqa.json ${MUSIQUE_DATA} > ${VAL_FILE}_misque/result.txt
echo "starting Ambig"
python ../src/ambigdoc_inference.py --input_dir ${AMBIG_DATA} --output_dir ${AMBIG_VAL_FILE} --max_samples ${MAX_SAMPLE}
python ../src/eval/eval_ambigdoc.py ${AMBIG_VAL_FILE}/processed_ambigqa.json > ${AMBIG_VAL_FILE}/result.txt