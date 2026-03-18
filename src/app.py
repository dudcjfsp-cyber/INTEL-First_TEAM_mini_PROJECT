"""
Streamlit 웹캠 데모 앱 (app.py)
분리수거 실시간 분류 - EfficientNet-B0 기반
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import timm
import cv2
from PIL import Image
from torchvision import transforms
import streamlit as st

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── 경로 설정 ───────────────────────────────────────────
MODEL_PATH = os.path.join("models", "best_model.pth")
CONFIG_PATH = os.path.join("models", "model_config.json")

DEFAULT_CLASSES = ["plastic", "can", "paper", "glass", "trash"]
CLASS_KO = {
    "plastic": "플라스틱",
    "can": "캔",
    "paper": "종이",
    "glass": "유리",
    "trash": "기타(재활용불가)"
}
CLASS_ICON = {
    "plastic": "🧴",
    "can": "🥫",
    "paper": "📄",
    "glass": "🍾",
    "trash": "🗑️"
}
RECYCLABLE = {"plastic": True, "can": True, "paper": True, "glass": True, "trash": False}


# ─── 페이지 설정 ─────────────────────────────────────────
st.set_page_config(
    page_title="재활용 분류 데모",
    page_icon="♻️",
    layout="wide"
)

# ─── 커스텀 CSS ──────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
    * { font-family: 'Noto Sans KR', sans-serif; }
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }
    .result-ok {
        background: linear-gradient(135deg, #d5f5e3, #a9dfbf);
        border: 2px solid #2ecc71;
        color: #1a7a4a;
    }
    .result-ng {
        background: linear-gradient(135deg, #fdedec, #f5b7b1);
        border: 2px solid #e74c3c;
        color: #922b21;
    }
    .class-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        background: #2c3e50;
        color: white;
        font-weight: bold;
        font-size: 1.3rem;
        margin: 0.5rem;
    }
    .conf-bar-container {
        background: #ecf0f1;
        border-radius: 8px;
        height: 18px;
        margin: 4px 0;
        overflow: hidden;
    }
    .conf-bar {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #2ecc71, #27ae60);
        transition: width 0.3s ease;
    }
    .sidebar-info {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── 모델 로드 (캐싱) ─────────────────────────────────────
@st.cache_resource
def load_everything():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config 로드
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "model_name": "efficientnet_b0",
            "num_classes": len(DEFAULT_CLASSES),
            "class_names": DEFAULT_CLASSES,
            "class_ko": CLASS_KO,
            "recyclable": RECYCLABLE,
        }

    # 모델 로드
    model = timm.create_model(
        config["model_name"],
        pretrained=not os.path.exists(MODEL_PATH),
        num_classes=config["num_classes"]
    )
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return model, transform, config, device


@torch.no_grad()
def predict(model, image: Image.Image, transform, config, device):
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    class_names = config["class_names"]
    top_idx = int(np.argmax(probs))
    top_class = class_names[top_idx]
    top_conf = float(probs[top_idx]) * 100

    return {
        "class_en": top_class,
        "class_ko": config.get("class_ko", CLASS_KO).get(top_class, top_class),
        "icon": CLASS_ICON.get(top_class, "📦"),
        "confidence": top_conf,
        "recyclable": config.get("recyclable", RECYCLABLE).get(top_class, False),
        "all_probs": {class_names[i]: float(probs[i]) * 100 for i in range(len(class_names))},
    }


# ─── 메인 UI ─────────────────────────────────────────────
def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>♻️ 재활용 분류 실시간 데모</h1>
        <p>웹캠 앞에 분리수거 대상을 비춰보세요</p>
    </div>
    """, unsafe_allow_html=True)

    # 모델 로드
    with st.spinner("모델 로딩 중..."):
        model, transform, config, device = load_everything()

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        confidence_threshold = st.slider("신뢰도 임계값 (%)", 0, 100, 50, 5)
        smoothing = st.slider("스무딩 프레임 수", 1, 10, 3)
        st.markdown("---")
        st.header("📋 분류 클래스")
        for cls in config["class_names"]:
            ko = config.get("class_ko", CLASS_KO).get(cls, cls)
            can = config.get("recyclable", RECYCLABLE).get(cls, False)
            icon = CLASS_ICON.get(cls, "📦")
            badge = "✅ 재활용 가능" if can else "❌ 재활용 불가"
            st.markdown(f"{icon} **{ko}** — {badge}")

        st.markdown("---")
        st.info(f"🖥️ 디바이스: `{'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}`")

    # 탭 구성
    tab_webcam, tab_upload = st.tabs(["📹 웹캠 실시간", "🖼️ 이미지 업로드"])

    # ── 웹캠 탭 ──
    with tab_webcam:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("웹캠 화면")
            run = st.toggle("🔴 웹캠 시작", value=False)
            frame_placeholder = st.empty()
            status_placeholder = st.empty()

        with col2:
            st.subheader("예측 결과")
            result_placeholder = st.empty()
            prob_placeholder = st.empty()

        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ 웹캠을 열 수 없습니다. 웹캠 연결 상태를 확인해주세요.")
            else:
                recent_preds = []
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ 프레임을 읽을 수 없습니다.")
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    result = predict(model, pil_img, transform, config, device)

                    # 스무딩
                    recent_preds.append(result["class_en"])
                    if len(recent_preds) > smoothing:
                        recent_preds.pop(0)
                    from collections import Counter
                    smoothed_class = Counter(recent_preds).most_common(1)[0][0]

                    # 프레임에 결과 오버레이
                    conf_color = (0, 200, 80) if result["recyclable"] else (220, 50, 50)
                    label = f"{result['class_ko']} {result['confidence']:.0f}%"
                    cv2.putText(frame_rgb, label, (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, conf_color, 3)
                    recyc_text = "RECYCLABLE" if result["recyclable"] else "NOT RECYCLABLE"
                    cv2.putText(frame_rgb, recyc_text, (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf_color, 2)

                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    # 결과 표시
                    if result["confidence"] >= confidence_threshold:
                        css_class = "result-ok" if result["recyclable"] else "result-ng"
                        recyc_label = "✅ 재활용 가능" if result["recyclable"] else "❌ 재활용 불가"
                        result_placeholder.markdown(f"""
                        <div class="result-box {css_class}">
                            <div style="font-size:2.5rem">{result['icon']}</div>
                            <div class="class-badge">{result['class_ko']}</div>
                            <div style="font-size:1.4rem; font-weight:bold; margin:0.5rem 0">{recyc_label}</div>
                            <div style="font-size:1.1rem">신뢰도: <b>{result['confidence']:.1f}%</b></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        result_placeholder.info(f"⏳ 신뢰도 부족 ({result['confidence']:.1f}% < {confidence_threshold}%)")

                    # 확률 분포
                    with prob_placeholder.container():
                        st.markdown("**전체 확률 분포**")
                        for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
                            ko = config.get("class_ko", CLASS_KO).get(cls, cls)
                            icon = CLASS_ICON.get(cls, "📦")
                            st.markdown(f"{icon} {ko}")
                            st.progress(int(prob))

                    time.sleep(0.05)

                cap.release()

    # ── 이미지 업로드 탭 ──
    with tab_upload:
        uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, caption="업로드된 이미지", use_container_width=True)
            with col2:
                with st.spinner("추론 중..."):
                    result = predict(model, img, transform, config, device)

                css_class = "result-ok" if result["recyclable"] else "result-ng"
                recyc_label = "✅ 재활용 가능" if result["recyclable"] else "❌ 재활용 불가"
                st.markdown(f"""
                <div class="result-box {css_class}">
                    <div style="font-size:3rem">{result['icon']}</div>
                    <div class="class-badge">{result['class_ko']}</div>
                    <div style="font-size:1.5rem; font-weight:bold; margin:0.5rem 0">{recyc_label}</div>
                    <div style="font-size:1.2rem">신뢰도: <b>{result['confidence']:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("**전체 확률 분포**")
                for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
                    ko = config.get("class_ko", CLASS_KO).get(cls, cls)
                    icon = CLASS_ICON.get(cls, "📦")
                    col_a, col_b = st.columns([2, 1])
                    col_a.markdown(f"{icon} {ko}")
                    col_b.markdown(f"**{prob:.1f}%**")
                    st.progress(int(prob))


if __name__ == "__main__":
    main()
