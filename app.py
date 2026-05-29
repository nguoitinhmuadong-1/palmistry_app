import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

st.set_page_config(
    page_title="Palmistry AI",
    page_icon="✋",
    layout="wide"
)

YOLO_MODEL_PATH = "palmistry_yolo_best.pt"
CNN_MODEL_PATH = "palmistry_cnn_quality.h5"

# =========================
# Load YOLO
# =========================
@st.cache_resource
def load_yolo_model():
    return YOLO(YOLO_MODEL_PATH)

# =========================
# Load CNN nếu được, lỗi thì bỏ qua
# =========================
@st.cache_resource
def load_cnn_model():
    try:
        from keras.models import load_model
        model = load_model(CNN_MODEL_PATH, compile=False)
        return model
    except Exception as e:
        return None

def check_line_quality(crop, cnn_model=None):
    """
    Nếu CNN load được thì dùng CNN.
    Nếu CNN lỗi thì dùng OpenCV để đánh giá rõ/mờ.
    """
    if crop is None or crop.size == 0:
        return "Không rõ", 0.0

    # Dùng CNN nếu load được
    if cnn_model is not None:
        try:
            resized = cv2.resize(crop, (224, 224))
            arr = resized.astype("float32") / 255.0
            arr = np.expand_dims(arr, axis=0)

            pred = cnn_model.predict(arr, verbose=0)
            class_id = int(np.argmax(pred))
            conf = float(np.max(pred))

            class_names = ["blur_line", "clear_line"]
            result = class_names[class_id]

            if result == "clear_line":
                return "Rõ", conf
            else:
                return "Mờ", conf
        except:
            pass

    # Fallback OpenCV
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if score >= 120:
        return "Rõ", min(score / 300, 1.0)
    elif score >= 60:
        return "Trung bình", min(score / 300, 1.0)
    else:
        return "Mờ", min(score / 300, 1.0)

def palmistry_comment(line_name, quality):
    comments = {
        "heart": {
            "vi": "Đường Tâm Đạo",
            "text": "liên quan đến cảm xúc, tình cảm và cách thể hiện cảm xúc."
        },
        "head": {
            "vi": "Đường Trí Đạo",
            "text": "liên quan đến tư duy, khả năng suy nghĩ và cách đưa ra quyết định."
        },
        "life": {
            "vi": "Đường Sinh Đạo",
            "text": "thường được xem là biểu tượng về sức sống và năng lượng cá nhân."
        },
        "fate": {
            "vi": "Đường Vận Mệnh",
            "text": "thường được liên hệ với định hướng, sự nghiệp và những thay đổi trong cuộc sống."
        }
    }

    info = comments.get(line_name, {
        "vi": line_name,
        "text": "là một đường chỉ tay được phát hiện trên lòng bàn tay."
    })

    if quality == "Rõ":
        q_comment = "Đường này khá rõ nên kết quả nhận diện có độ tin cậy tốt hơn."
    elif quality == "Trung bình":
        q_comment = "Đường này có độ rõ trung bình, kết quả chỉ nên xem là tham khảo."
    else:
        q_comment = "Đường này khá mờ, ảnh có thể chưa đủ rõ hoặc đường chỉ tay không nổi bật."

    return info["vi"], f"{info['vi']} {info['text']} {q_comment}"

def draw_label(img, text, x, y):
    cv2.putText(
        img,
        text,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Cài đặt")

conf_threshold = st.sidebar.slider(
    "Ngưỡng nhận diện YOLO",
    min_value=0.05,
    max_value=0.80,
    value=0.25,
    step=0.05
)

st.sidebar.info(
    "Giảm ngưỡng nếu model không phát hiện đường chỉ tay. "
    "Tăng ngưỡng nếu model nhận nhầm quá nhiều."
)

st.sidebar.subheader("File model cần có")
st.sidebar.code(
    "palmistry_yolo_best.pt\npalmistry_cnn_quality.h5"
)

# =========================
# Main UI
# =========================
st.markdown(
    """
    <h1 style='text-align:center;'>✋ Palmistry AI</h1>
    <h3 style='text-align:center;'>Nhận diện đường chỉ tay bằng YOLO và đánh giá rõ/mờ</h3>
    """,
    unsafe_allow_html=True
)

if not os.path.exists(YOLO_MODEL_PATH):
    st.error("Không tìm thấy file palmistry_yolo_best.pt")
    st.stop()

yolo_model = load_yolo_model()
cnn_model = load_cnn_model()

if cnn_model is None:
    st.warning("CNN không load được do khác phiên bản Keras. App sẽ dùng YOLO + OpenCV để đánh giá rõ/mờ.")
else:
    st.success("Đã load YOLO và CNN thành công.")

tab1, tab2 = st.tabs(["📤 Upload ảnh", "📷 Chụp camera"])

image_file = None

with tab1:
    image_file = st.file_uploader(
        "Tải ảnh lòng bàn tay",
        type=["jpg", "jpeg", "png"]
    )

with tab2:
    camera_file = st.camera_input("Chụp ảnh lòng bàn tay")
    if camera_file is not None:
        image_file = camera_file

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.subheader("Ảnh gốc")
    st.image(image, use_container_width=True)

    results = yolo_model.predict(
        source=img_bgr,
        conf=conf_threshold,
        verbose=False
    )

    output_img = img_np.copy()
    detected_results = []

    for result in results:
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            line_name = yolo_model.names[cls_id]

            crop = img_bgr[y1:y2, x1:x2]
            quality, quality_conf = check_line_quality(crop, cnn_model)

            vi_name, comment = palmistry_comment(line_name, quality)

            detected_results.append({
                "line": line_name,
                "vi_name": vi_name,
                "quality": quality,
                "yolo_conf": conf,
                "quality_conf": quality_conf,
                "comment": comment
            })

            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{line_name} - {quality}"
            draw_label(output_img, label, x1, y1)

    st.subheader("Kết quả nhận diện")
    st.image(output_img, use_container_width=True)

    if len(detected_results) == 0:
        st.warning("Không phát hiện được đường chỉ tay. Anh thử giảm ngưỡng YOLO hoặc dùng ảnh rõ hơn.")
    else:
        st.subheader("Phân tích kết quả")

        for item in detected_results:
            with st.container():
                st.markdown(f"### {item['vi_name']}")
                st.write(f"**Tên class:** `{item['line']}`")
                st.write(f"**Độ tin cậy YOLO:** `{item['yolo_conf']:.2f}`")
                st.write(f"**Độ rõ:** `{item['quality']}`")
                st.write(item["comment"])
                st.divider()

        st.info(
            "Lưu ý: Kết quả palmistry chỉ mang tính tham khảo và giải trí, "
            "không phải kết luận khoa học hay dự đoán chắc chắn."
        )
else:
    st.info("Anh hãy upload ảnh hoặc chụp ảnh lòng bàn tay để bắt đầu.")