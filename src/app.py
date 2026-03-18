"""
Streamlit 웹캠 데모 앱 (app.py)
분리수거 실시간 분류 - Dual-Head MobileNetV3 기반
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
from PIL import Image
import streamlit as st

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 추론 함수 임포트
from src.inference import load_config, load_model, predict, CLASS_KO

CLASS_ICON = {
    "plastic": "🧴",
    "can": "🥫",
    "paper": "📄",
    "glass": "🍾",
    "trash": "🗑️"
}

# ─── 페이지 설정 ─────────────────────────────────────────
st.set_page_config(
    page_title="재활용 듀얼헤드 데모",
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
        margin: 0.2rem;
    }
    .contam-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        background: #e67e22;
        color: white;
        font-weight: bold;
        font-size: 1.3rem;
        margin: 0.2rem;
    }
    .clean-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        background: #3498db;
        color: white;
        font-weight: bold;
        font-size: 1.3rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── 모델 로드 (캐싱) ─────────────────────────────────────
@st.cache_resource
def load_everything():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    model = load_model(config, device)
    return model, config, device

# ─── 메인 UI ─────────────────────────────────────────────
def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>♻️ 재활용 듀얼헤드(재질+오염) 실시간 데모</h1>
        <p>웹캠 앞에 분리수거 대상을 비춰보세요</p>
    </div>
    """, unsafe_allow_html=True)

    # 모델 로드
    with st.spinner("듀얼헤드 모델 로딩 중..."):
        model, config, device = load_everything()

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        confidence_threshold = st.slider("재질 신뢰도 임계값 (%)", 0, 100, 50, 5)
        smoothing = st.slider("스무딩 프레임 수", 1, 10, 3)
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

        with col2:
            st.subheader("듀얼 헤드 예측 결과")
            result_placeholder = st.empty()
            prob_placeholder = st.empty()

        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ 웹캠을 열 수 없습니다.")
            else:
                recent_preds = []
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ 프레임을 읽을 수 없습니다.")
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    
                    # 듀얼헤드 추론
                    result = predict(model, pil_img, config, device)
                    mat = result["material"]
                    contam = result["contamination"]

                    # 스무딩
                    recent_preds.append(mat["class_en"])
                    if len(recent_preds) > smoothing:
                        recent_preds.pop(0)
                    from collections import Counter
                    smoothed_class = Counter(recent_preds).most_common(1)[0][0]

                    # 프레임에 결과 오버레이
                    conf_color = (0, 200, 80) if result["final_recyclable"] else (220, 50, 50)
                    label = f"{mat['class_ko']} {mat['confidence']:.0f}%"
                    contam_label = f"Contam: {contam['is_contaminated']}"
                    
                    cv2.putText(frame_rgb, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, conf_color, 3)
                    cv2.putText(frame_rgb, contam_label, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,165,0), 2)
                    
                    recyc_text = "RECYCLABLE" if result["final_recyclable"] else "NOT RECYCLABLE"
                    cv2.putText(frame_rgb, recyc_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf_color, 2)

                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    # 결과 표시
                    if mat["confidence"] >= confidence_threshold:
                        css_class = "result-ok" if result["final_recyclable"] else "result-ng"
                        recyc_label = "✅ 최종: 재활용 가능" if result["final_recyclable"] else "❌ 최종: 재활용 불가"
                        icon = CLASS_ICON.get(mat["class_en"], "📦")
                        
                        contam_css = "contam-badge" if contam["is_contaminated"] else "clean-badge"

                        result_placeholder.markdown(f"""
                        <div class="result-box {css_class}">
                            <div style="font-size:2.5rem">{icon}</div>
                            <div>
                                <span class="class-badge">재질: {mat['class_ko']} ({mat['confidence']:.1f}%)</span>
                            </div>
                            <div>
                                <span class="{contam_css}">상태: {contam['status_ko']} ({contam['confidence']:.1f}%)</span>
                            </div>
                            <div style="font-size:1.4rem; font-weight:bold; margin:0.5rem 0">{recyc_label}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        result_placeholder.info(f"⏳ 대기 중 (신뢰도 부족: {mat['confidence']:.1f}%)")

                    # 확률 분포
                    with prob_placeholder.container():
                        st.markdown("**재질 확률 분포**")
                        for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
                            ko = config.get("class_ko", CLASS_KO).get(cls, cls)
                            ico = CLASS_ICON.get(cls, "📦")
                            st.markdown(f"{ico} {ko}")
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
                    result = predict(model, img, config, device)
                    mat = result["material"]
                    contam = result["contamination"]

                css_class = "result-ok" if result["final_recyclable"] else "result-ng"
                recyc_label = "✅ 최종: 재활용 가능" if result["final_recyclable"] else "❌ 최종: 재활용 불가"
                icon = CLASS_ICON.get(mat["class_en"], "📦")
                
                contam_css = "contam-badge" if contam["is_contaminated"] else "clean-badge"

                st.markdown(f"""
                <div class="result-box {css_class}">
                    <div style="font-size:3rem">{icon}</div>
                    <div>
                        <span class="class-badge">재질: {mat['class_ko']} ({mat['confidence']:.1f}%)</span>
                    </div>
                    <div>
                        <span class="{contam_css}">상태: {contam['status_ko']} ({contam['confidence']:.1f}%)</span>
                    </div>
                    <div style="font-size:1.5rem; font-weight:bold; margin:0.5rem 0">{recyc_label}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("**재질 확률 분포**")
                for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
                    ko = config.get("class_ko", CLASS_KO).get(cls, cls)
                    ico = CLASS_ICON.get(cls, "📦")
                    col_a, col_b = st.columns([2, 1])
                    col_a.markdown(f"{ico} {ko}")
                    col_b.markdown(f"**{prob:.1f}%**")
                    st.progress(int(prob))


if __name__ == "__main__":
    main()
