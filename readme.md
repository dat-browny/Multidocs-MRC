# VNPT Multi-Document Machine Reading Comprehension
Mô hình đọc hiểu văn bản tiếng việt và trả lời câu hỏi

[Báo cáo kết quả thử nghiệm](https://docs.google.com/document/d/1Hbj6XPuBuyoyHfPa_VD7cG1nOn2fYgleMA-iY4rNoN4/edit?usp=sharing)

## Nội Dung
1. [Cài đặt](#setup) <br>
2. [Đào tạo model](#train_model) <br>
    2.1 [Dữ liệu](#training_data) <br>
    2.2 [Đào tạo model](#train_model_script) <br>
    2.3 [Đánh giá model](#evaluate_model) <br>
3. [Sử dụng Python Package](#inference) <br>
4. [Docker](#docker) <br>
    4.1 [Build and Run Docker](#build_and_run_docker) <br>
    4.2 [Service](#service) <br>


## 1. Cài đặt <a name="setup"></a>
```bash
pip install -r requirements.txt 
pip install -e .
```

## 2. Đào tạo model <a name="train_model"></a>
### 2.1 Dữ liệu <a name="training_data"></a>
Các file dữ liệu train và test theo định dạng `Squad2.0`

### 2.2 Đào tạo model <a name="train_model_script"></a>
Thay đổi các tham số model_path, data_path, .. trong file `./run_train.sh`, sau đó mở terminal và chạy đoạn mã sau
```bash
./run_train.sh
```
Để biết thêm về các tham số, mở terminal và chạy `python multi_document_mrc/run_train.py --help`

### 2.3 Đánh giá model <a name="evaluate_model"></a>
Thay đổi các tham số model_path, data_path, .. trong file `./run_evaluate.sh`, sau đó mở terminal và chạy đoạn mã sau
```bash
./run_evaluate.sh
```


## 3. Sử dụng Python Package <a name="inference"></a>


## 4. Docker <a name="docker"></a>

### 4.1 Build and Run Docker <a name="build_and_run_docker"></a>
**Build:** <br>
```bash
docker build -t hub.vnpt.vn/smartbot-prod/vnpt_multi_document_mrc:latest -f Dockerfile .
```

**Run on CPU:** <br>
Run: <br>
```bash
docker run --cpus="4" -p 5000:5000 --network host \
    -e REDIS_HOST='127.0.0.1' \
    -e REDIS_PORT='6379' \
    -e REDIS_DB='2' \
    -e REDIS_PW='' \
    -it --rm hub.vnpt.vn/smartbot-prod/vnpt_multi_document_mrc:latest
```
`REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PW` là địa chỉ và các config của redis database bên ngoài
Nếu không set các biến môi trường trên, Model sẽ tự động sử dụng Local Redis DB trong container

**Run on GPU:** <br>
Run: <br>
```bash
docker run --gpus all -p 5000:5000 --network host \
    -e REDIS_HOST='127.0.0.1' \
    -e REDIS_PORT='6379' \
    -e REDIS_DB='2' \
    -e REDIS_PW='' \
    -it --rm hub.vnpt.vn/smartbot-prod/vnpt_multi_document_mrc:latest
```

### 4.2 Service <a name="service"></a>

```python
import requests

data = {
    "query":"Thủ đô của việt nam là gì",
    "knowledge_retrieval":[
        {
            "passage_content":"Hà Nội, thủ đô của Việt Nam, nổi tiếng với kiến trúc trăm tuổi và nền văn hóa phong phú với sự ảnh hưởng của khu vực Đông Nam Á, Trung Quốc và Pháp. Trung tâm thành phố là Khu phố cổ nhộn nhịp, nơi các con phố hẹp được mang tên \"hàng\".",
            "passage_title":"Hà Nội"
        },
        {
            "passage_content": "Thủ đô Việt Nam hiện nay là thành phố Hà Nội. Sau đây là danh sách các kinh đô/thủ đô – hiểu theo nghĩa rộng – là các trung tâm chính trị của chính thể nhà nước trong lịch sử Việt Nam, và cả của các vương quốc cổ/cựu quốc gia từng tồn tại trên lãnh thổ Việt Nam ngày nay.",
            "passage_title":"Thủ đô Việt Nam"
        }
    ],
    "knowledge_search":[
        {
            "passage_content": "Việt Nam, quốc hiệu chính thức là Cộng hòa Xã hội chủ nghĩa Việt Nam, là một quốc gia nằm ở cực Đông của bán đảo Đông Dương thuộc khu vực Đông Nam Á",
            "passage_title": ""
        }
    ]
}

res = requests.post(
    url="http://localhost:5000/knowledge_grounded_response",
    json={
        "sender_id":"123",
        "bot_id": "123",
        "data": data
})
print(res.json())
```
