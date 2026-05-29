# Palmistry AI App

App Streamlit nhận diện đường chỉ tay bằng YOLO và đánh giá rõ/mờ bằng CNN.

## File cần có

- `app.py`
- `palmistry_yolo_best.pt`
- `palmistry_cnn_quality.h5`
- `requirements.txt`

## Chạy local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Kết quả app

- Nhận diện: heart, head, life, fate.
- CNN phân loại: clear_line / blur_line.
- Hiển thị ảnh kết quả và nhận xét palmistry dạng tham khảo.
