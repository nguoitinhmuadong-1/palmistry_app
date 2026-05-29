import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AD Palmistry",
    page_icon="✋",
    layout="wide"
)

YOLO_MODEL_PATH = "palmistry_yolo_best.pt"
CNN_MODEL_PATH = "palmistry_cnn_quality.h5"

# =========================
# CSS UI
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e1b4b 100%);
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .main-title {
        text-align: center;
        font-size: 58px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 0px;
        background: linear-gradient(90deg, #fde68a, #f9a8d4, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-title {
        text-align: center;
        font-size: 22px;
        color: #dbeafe;
        margin-bottom: 30px;
    }

    .hero-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 24px;
        padding: 28px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        margin-bottom: 24px;
    }

    .feature-card {
        background: rgba(30, 41, 59, 0.72);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 22px;
        height: 100%;
    }

    .result-card {
        background: rgba(15, 23, 42, 0.82);
        border-left: 5px solid #facc15;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }

    .line-name {
        font-size: 24px;
        font-weight: 700;
        color: #fde68a;
        margin-bottom: 6px;
    }

    .metric-pill {
        display: inline-block;
        padding: 6px 12px;
        margin: 4px 6px 6px 0px;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.18);
        border: 1px solid rgba(147,197,253,0.35);
        color: #dbeafe;
        font-size: 14px;
    }

    .note-box {
        background: rgba(250, 204, 21, 0.12);
        border: 1px solid rgba(250, 204, 21, 0.25);
        border-radius: 16px;
        padding: 16px;
        color: #fef3c7;
        margin-top: 20px;
    }

    .upload-box {
        background: rgba(30, 41, 59, 0.72);
        border: 1px dashed rgba(147,197,253,0.45);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 18px;
    }

    .small-muted {
        color: #cbd5e1;
        font-size: 15px;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.35);
        padding: 16px;
        border-radius: 15px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 14px;
        color: #e5e7eb;
        padding: 12px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_yolo_model():
    return YOLO(YOLO_MODEL_PATH)

@st.cache_resource
def load_cnn_model():
    try:
        from keras.models import load_model
        model = load_model(CNN_MODEL_PATH, compile=False)
        return model
    except Exception:
        return None

# =========================
# PROCESSING FUNCTIONS
# =========================
def check_line_quality(crop, cnn_model=None):
    if crop is None or crop.size == 0:
        return "Không rõ", 0.0

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
        except Exception:
            pass

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
            "icon": "💗",
            "meaning": "liên quan đến cảm xúc, tình cảm và cách thể hiện cảm xúc."
        },
        "head": {
            "vi": "Đường Trí Đạo",
            "icon": "🧠",
            "meaning": "liên quan đến tư duy, khả năng suy nghĩ và cách đưa ra quyết định."
        },
        "life": {
            "vi": "Đường Sinh Đạo",
            "icon": "🌿",
            "meaning": "thường được xem là biểu tượng về sức sống và năng lượng cá nhân."
        },
        "fate": {
            "vi": "Đường Vận Mệnh",
            "icon": "✨",
            "meaning": "thường được liên hệ với định hướng, sự nghiệp và những thay đổi trong cuộc sống."
        }
    }

    info = comments.get(line_name, {
        "vi": line_name,
        "icon": "✋",
        "meaning": "là một đường chỉ tay được phát hiện trên lòng bàn tay."
    })

    if quality == "Rõ":
        q_comment = "Đường này khá rõ nên kết quả nhận diện có độ tin cậy tốt hơn."
    elif quality == "Trung bình":
        q_comment = "Đường này có độ rõ trung bình, kết quả chỉ nên xem là tham khảo."
    else:
        q_comment = "Đường này khá mờ, ảnh có thể chưa đủ sáng hoặc đường chỉ tay không nổi bật."

    return info["vi"], info["icon"], f"{info['vi']} {info['meaning']} {q_comment}"

def draw_label(img, text, x, y):
    cv2.rectangle(img, (x, max(y - 34, 0)), (x + len(text) * 12 + 12, max(y - 5, 25)), (15, 23, 42), -1)
    cv2.putText(
        img,
        text,
        (x + 6, max(y - 12, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## ⚙️ Cài đặt")
st.sidebar.markdown("Điều chỉnh độ nhạy khi nhận diện đường chỉ tay.")

conf_threshold = st.sidebar.slider(
    "Ngưỡng nhận diện",
    min_value=0.05,
    max_value=0.80,
    value=0.25,
    step=0.05
)

st.sidebar.markdown(
    """
    <div class="note-box">
    <b>Gợi ý:</b><br>
    Nếu app không phát hiện đường chỉ tay, hãy giảm ngưỡng xuống 0.15–0.20.
    Nếu nhận nhầm quá nhiều, tăng lên 0.30–0.40.
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("## 📌 Hướng dẫn chụp ảnh")
st.sidebar.markdown(
    """
    - Chụp lòng bàn tay rõ nét  
    - Đặt tay ở nơi đủ sáng  
    - Tránh ảnh quá xa hoặc bị nghiêng  
    - Nên để lòng bàn tay chiếm phần lớn ảnh  
    """
)

# =========================
# MAIN HEADER
# =========================
st.markdown("<div class='main-title'>✋ AD Palmistry</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Nhận diện các đường chỉ tay bằng YOLO và phân tích kết quả theo phong cách palmistry</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-card">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px;">
            <div class="feature-card">
                <h3>🔍 Nhận diện thông minh</h3>
                <p class="small-muted">Phát hiện các đường chỉ tay chính như Tâm Đạo, Trí Đạo, Sinh Đạo và Vận Mệnh.</p>
            </div>
            <div class="feature-card">
                <h3>📷 Upload hoặc Camera</h3>
                <p class="small-muted">Hỗ trợ tải ảnh từ máy hoặc chụp trực tiếp bằng camera.</p>
            </div>
            <div class="feature-card">
                <h3>🌙 Phân tích tham khảo</h3>
                <p class="small-muted">Đưa ra nhận xét palmistry dạng giải trí, không phải kết luận khoa học.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# CHECK MODEL
# =========================
if not os.path.exists(YOLO_MODEL_PATH):
    st.error("Không tìm thấy file model YOLO. Vui lòng kiểm tra file palmistry_yolo_best.pt.")
    st.stop()

yolo_model = load_yolo_model()
cnn_model = load_cnn_model()

# =========================
# INPUT AREA
# =========================
st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
st.markdown("## 🖼️ Chọn ảnh lòng bàn tay")

tab1, tab2 = st.tabs(["📤 Tải ảnh lên", "📷 Chụp bằng camera"])

image_file = None

with tab1:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    image_file = st.file_uploader(
        "Chọn ảnh lòng bàn tay",
        type=["jpg", "jpeg", "png"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    camera_file = st.camera_input("Chụp ảnh lòng bàn tay")
    if camera_file is not None:
        image_file = camera_file
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PREDICTION
# =========================
if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Ảnh gốc")
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
            vi_name, icon, comment = palmistry_comment(line_name, quality)

            detected_results.append({
                "line": line_name,
                "vi_name": vi_name,
                "icon": icon,
                "quality": quality,
                "yolo_conf": conf,
                "quality_conf": quality_conf,
                "comment": comment
            })

            cv2.rectangle(output_img, (x1, y1), (x2, y2), (34, 197, 94), 3)
            draw_label(output_img, f"{line_name} - {quality}", x1, y1)

    with col2:
        st.markdown("### Ảnh sau nhận diện")
        st.image(output_img, use_container_width=True)

    st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
    st.markdown("## 🔮 Kết quả phân tích")

    if len(detected_results) == 0:
        st.warning("Không phát hiện được đường chỉ tay. Anh thử giảm ngưỡng nhận diện hoặc dùng ảnh rõ hơn.")
    else:
        for item in detected_results:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="line-name">{item['icon']} {item['vi_name']}</div>
                    <span class="metric-pill">Class: {item['line']}</span>
                    <span class="metric-pill">Độ tin cậy YOLO: {item['yolo_conf']:.2f}</span>
                    <span class="metric-pill">Độ rõ: {item['quality']}</span>
                    <p style="margin-top: 12px; color: #e5e7eb;">{item['comment']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div class="note-box">
            <b>Lưu ý:</b> Kết quả palmistry chỉ mang tính tham khảo và giải trí. 
            App không đưa ra kết luận khoa học hay dự đoán chắc chắn về tương lai.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="hero-card">
            <h2>👋 Sẵn sàng bắt đầu</h2>
            <p class="small-muted">
            Anh hãy tải lên một ảnh lòng bàn tay hoặc chụp ảnh trực tiếp bằng camera.
            App sẽ tự động nhận diện các đường chỉ tay và hiển thị kết quả phân tích.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )