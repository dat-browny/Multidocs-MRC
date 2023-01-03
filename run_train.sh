#!/bin/bash 

python3 multi_document_mrc/run_train.py \
    --model_name_or_path /data/language_model/PhoBert/phobert-base \
    --model_architecture roberta-qa \
    --output_dir mounts/models/phobert-base-qa-lr5e5-bs16-e10 \
    --train_file /data/data/labeled_vietnamese/MRC_VLSP/v2_train_ViQuAD.json \
    --validation_file /data/data/labeled_vietnamese/MRC_VLSP/v1_dev_ViQuAD.json \
    --do_train \
    --do_eval \
    --learning_rate 0.00005 \
    --max_seq_length 256 \
    --doc_stride 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 10

