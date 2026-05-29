import os
import tempfile
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# =========================
# CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="Palmistry AI",
    page_icon="✋",
    layout="wide",
)

YOLO_MODEL_PATH = "palmistry_yolo_best.pt"
CNN_MODEL_PATH = "palmistry_cnn_quality.h5"
CNN_IMG_SIZE = 224

LINE_VI = {
    "heart": "Đường Tâm Đạo",
    "head": "Đường Trí Đạo",
    "life": "Đường Sinh Đạo",
    "fate": "Đường Vận Mệnh",
}

LINE_MEANING = {
    "heart": "liên quan đến cảm xúc, tình cảm và cách thể hiện cảm xúc.",
    "head": "liên quan đến tư duy, khả năng suy nghĩ và định hướng lý trí.",
    "life": "thường được xem là biểu tượng về sức sống, năng lượng và sự ổn định.",
    "fate": "thường được liên hệ với định hướng, mục tiêu và sự thay đổi trong cuộc sống.",
}

CNN_CLASS_NAMES = ["blur_line", "clear_line"]
QUALITY_VI = {
    "blur_line": "Mờ",
    "clear_line": "Rõ",
}

# Màu vẽ khung BGR cho OpenCV
BOX_COLORS = {
    "heart": (255, 80, 80),
    "head": (80, 180, 255),
    "life": (80, 220, 120),
    "fate": (200, 120, 255),
}

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_yolo_model():
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy file {YOLO_MODEL_PATH}")
    return YOLO(YOLO_MODEL_PATH)

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy file {CNN_MODEL_PATH}")
    return load_model(CNN_MODEL_PATH)

# =========================
# HÀM XỬ LÝ
# =========================
def predict_quality(cnn_model, crop_bgr: np.ndarray) -> Tuple[str, float]:
    """Dự đoán đường chỉ tay rõ/mờ bằng CNN."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0.0

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (CNN_IMG_SIZE, CNN_IMG_SIZE))
    arr = img_to_array(crop_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = cnn_model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(pred))
    conf = float(np.max(pred))

    if idx < len(CNN_CLASS_NAMES):
        return CNN_CLASS_NAMES[idx], conf
    return "unknown", conf


def analyze_image(image_pil: Image.Image, yolo_model, cnn_model, conf_threshold: float):
    """YOLO phát hiện đường chỉ tay, CNN đánh giá rõ/mờ."""
    image_rgb = np.array(image_pil.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    annotated = image_rgb.copy()

    results = yolo_model.predict(image_rgb, conf=conf_threshold, verbose=False)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            yolo_conf = float(box.conf[0])
            yolo_class = yolo_model.names.get(cls_id, str(cls_id))

            h, w = image_bgr.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            crop_bgr = image_bgr[y1:y2, x1:x2]
            quality_class, quality_conf = predict_quality(cnn_model, crop_bgr)

            detections.append({
                "line": yolo_class,
                "line_vi": LINE_VI.get(yolo_class, yolo_class),
                "yolo_conf": yolo_conf,
                "quality": quality_class,
                "quality_vi": QUALITY_VI.get(quality_class, "Không xác định"),
                "quality_conf": quality_conf,
                "box": (x1, y1, x2, y2),
                "crop_rgb": cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
            })

            color_bgr = BOX_COLORS.get(yolo_class, (0, 255, 0))
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_rgb, 3)

            label = f"{yolo_class} | {QUALITY_VI.get(quality_class, quality_class)}"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color_rgb,
                2,
                cv2.LINE_AA,
            )

    return annotated, detections


def build_summary(detections: List[Dict]) -> str:
    if not detections:
        return "Chưa phát hiện được đường chỉ tay rõ ràng. Anh nên thử ảnh sáng hơn, lòng bàn tay đặt chính giữa và chiếm phần lớn khung hình."

    lines = []
    found = {d["line"] for d in detections}

    for key in ["heart", "head", "life", "fate"]:
        items = [d for d in detections if d["line"] == key]
        if not items:
            lines.append(f"- **{LINE_VI.get(key, key)}**: Chưa phát hiện rõ.")
            continue

        best = max(items, key=lambda x: x["yolo_conf"])
        meaning = LINE_MEANING.get(key, "mang ý nghĩa tham khảo trong palmistry.")
        lines.append(
            f"- **{best['line_vi']}**: Đã phát hiện, độ rõ: **{best['quality_vi']}**. "
            f"Theo palmistry, đường này {meaning}"
        )

    return "\n".join(lines)

# =========================
# GIAO DIỆN
# =========================
st.markdown(
    """
    <h1 style='text-align:center;'>✋ Palmistry AI</h1>
    <p style='text-align:center; font-size:18px;'>Nhận diện đường chỉ tay bằng YOLO và đánh giá rõ/mờ bằng CNN</p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Cài đặt")
    conf_threshold = st.slider("Ngưỡng nhận diện YOLO", 0.05, 0.90, 0.25, 0.05)
    st.info("Giảm ngưỡng nếu model không phát hiện đường chỉ tay. Tăng ngưỡng nếu model nhận nhầm quá nhiều.")

    st.markdown("### File model cần có")
    st.code("palmistry_yolo_best.pt\npalmistry_cnn_quality.h5")

try:
    yolo_model = load_yolo_model()
    cnn_model = load_cnn_model()
except Exception as e:
    st.error(f"Không load được model: {e}")
    st.stop()

input_mode = st.radio(
    "Chọn cách nhập ảnh:",
    ["Tải ảnh từ máy", "Chụp bằng camera"],
    horizontal=True,
)

uploaded_image = None

if input_mode == "Tải ảnh từ máy":
    uploaded_image = st.file_uploader(
        "Tải ảnh lòng bàn tay",
        type=["jpg", "jpeg", "png"],
    )
else:
    uploaded_image = st.camera_input("Chụp ảnh lòng bàn tay")

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ảnh gốc")
        st.image(image, use_container_width=True)

    with st.spinner("Đang phân tích ảnh..."):
        annotated, detections = analyze_image(image, yolo_model, cnn_model, conf_threshold)

    with col2:
        st.subheader("Kết quả nhận diện")
        st.image(annotated, use_container_width=True)

    st.markdown("---")
    st.subheader("📌 Chi tiết kết quả")

    if len(detections) == 0:
        st.warning("Không phát hiện được đường chỉ tay nào. Hãy thử ảnh rõ hơn hoặc giảm ngưỡng nhận diện YOLO ở thanh bên trái.")
    else:
        for i, d in enumerate(detections, 1):
            with st.expander(f"{i}. {d['line_vi']} - {d['quality_vi']} | YOLO: {d['yolo_conf']:.2f}"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(d["crop_rgb"], caption="Vùng đường chỉ tay", use_container_width=True)
                with c2:
                    st.write(f"**Loại đường:** {d['line_vi']}")
                    st.write(f"**Độ tin cậy YOLO:** {d['yolo_conf']:.2f}")
                    st.write(f"**Đánh giá CNN:** {d['quality_vi']}")
                    st.write(f"**Độ tin cậy CNN:** {d['quality_conf']:.2f}")

    st.subheader("🔮 Nhận xét palmistry")
    st.markdown(build_summary(detections))

    st.warning("Lưu ý: Kết quả chỉ mang tính tham khảo và giải trí, không phải dự đoán khoa học.")
else:
    st.info("Anh hãy tải ảnh lòng bàn tay hoặc chụp ảnh trực tiếp để bắt đầu.")
