# Multi-Document Machine Reading Comprehension
Mô hình đọc hiểu văn bản tiếng việt và trả lời câu hỏi


## Nội Dung
1. [Cài đặt](#setup) <br>
2. [Đào tạo model](#train_model) <br>
    2.1 [Dữ liệu](#training_data) <br>
    2.2 [Đào tạo model](#train_model_script) <br>
    2.3 [Đánh giá model](#evaluate_model) <br>

## 1. Cài đặt <a name="setup"></a>
```bash
pip install -r requirements.txt 
pip install -e .
```

## 2. Đào tạo model <a name="train_model"></a>
### 2.1 Dữ liệu <a name="training_data"></a>
Các file dữ liệu train và test theo định dạng `Squad2.0`

### 2.2 Đào tạo model <a name="train_model_script"></a>
#### 2.2.1. Model MRC
Thay đổi các tham số model_path, data_path, .. trong file `./run_train.sh`, sau đó mở terminal và chạy đoạn mã sau
```bash
./run_train.sh
```
Để biết thêm về các tham số, mở terminal và chạy `python multi_document_mrc/run_train.py --help`
#### 2.2.2. Model Refection
Ban đầu, huấn luyện mô hình MRC từ pretrain PhoBert base
```bash
python3 run_mrc.py \
    --model_name_or_path vinai/phobert-base \
    --model_architecture phobert-qa-mrc-block \
    --output_dir ./phobert-base-mrc-lr5e5-bs32-e15 \
    --train_file ./MRC_VLSP/v2_train_ViQuAD_new.json \
    --do_train \
    --learning_rate 5e-5 \
    --max_seq_length 256 \
    --doc_stride 128  \
    --per_device_train_batch_size 16 \
    --num_train_epochs 15 \
    --save_total_limit 1
```
Tiếp theo, dùng mô hình MRC để huấn luyện mô hình Reflection, gồm hai bước: tạo data training mô hình Reflection và training mô hình Reflection.
```bashion
!python3 inference_reflection_data.py \
    --model_name_or_path vinai/phobert-base \
    --model_architecture phobert-qa-reflection-block \
    --output_dir ./ \
    --train_file ./Multidocs-MRC/MRC_VLSP/v1_dev_ViQuAD.json \
    --validation_file ./Multidocs-MRC/MRC_VLSP/v1_dev_ViQuAD.json \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --max_seq_length 256 \
    --doc_stride 128  \
```
```bash
!python3 run_reflection.py \
    --model_name_or_path vinai/phobert-base \
    --model_architecture phobert-qa-reflection-block \
    --output_dir ./phobert-base-qa-lr5-bs16-reflection \
    --train_dir ./train \
    --validation_dir ./eval \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --max_seq_length 256 \
    --doc_stride 128  \
    --per_device_train_batch_size 16 \
    --num_train_epochs 1
```

### 2.3 Đánh giá model <a name="evaluate_model"></a>
#### 2.3.1 Model MRC
Thay đổi các tham số model_path, data_path, .. trong file `./run_evaluate.sh`, sau đó mở terminal và chạy đoạn mã sau
```bash
./run_evaluate.sh
```
#### 2.3.2 Model Reflection
```bash
!python3 run_mrc.py \
    --model_name_or_path ./phobert-base-mrc-lr5-bs32-ep15 \
    --model_architecture phobert-qa-mrc-block \
    --reflection_path ./phobert-base-qa-lr5-bs16-reflection \
    --output_dir ./phobert-base-mrc-lr5-bs32-ep15 \
    --validation_file ./Multidocs-MRC/MRC_VLSP/v2_dev_ViQuAD.json \
    --version_2_with_negative \
    --do_eval \
    --learning_rate 5e-5 \
    --max_seq_length 256 \
    --doc_stride 128  \
    --per_device_train_batch_size 32 \
    --num_train_epochs 1 \
    --save_total_limit 1 \
    --overwrite_output_dir
```
