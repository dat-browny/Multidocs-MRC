#!/bin/bash 

python3 multi_document_mrc/run_train.py \
    --model_name_or_path /Users/brownyeyes/.cache/huggingface/hub/models--vinai--phobert-base/snapshots/a47f76f1ac0b6b89f82d15dbe9829935a0014c38 \
    --model_architecture phobert-qa \
    --output_dir ~/Downloads/phobert-base-qa-lr5e5-bs16-e10 \
    --train_file /Users/brownyeyes/Project/Multidoc-MRC/MRC_VLSP/v2_train_ViQuAD.json \
    --validation_file /Users/brownyeyes/Project/Multidoc-MRC/MRC_VLSP/v1_dev_ViQuAD.json \
    --do_train \
    --do_eval \
    --learning_rate 0.00005 \
    --max_seq_length 256 \
    --doc_stride 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 10

