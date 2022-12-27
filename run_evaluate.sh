#!/bin/bash 

python multi_document_mrc/run_train.py \
    --model_name_or_path mounts/models/test-xlm-roberta-base-e20 \
    --model_architecture vi-huggingface-qa \
    --output_dir mounts/models/test-xlm-roberta-base-e20 \
    --validation_file /data/data/labeled_vietnamese/MRC_VLSP/v1_test_ViQuAD.json \
    --do_eval \
    --per_device_eval_batch_size 16 \
    --max_answer_length 128 \
    --max_seq_length 384

