"""
Streamlit 웹캠 데모 앱 (app.py)
분리수거 실시간 분류 - 2-Stage Pipeline (YOLOv8 + Dual-Head MobileNetV3)
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

from src.inference import load_config, load_models, predict

CLASS_ICON = {
    "plastic": "🧴",
    "can": "🥫",
    "paper": "📄",
    "glass": "🍾",
    "trash": "🗑️"
}

# ─── 페이지 설정 ─────────────────────────────────────────
st.set_page_config(
    page_title="재활용 2-Stage 파이프라인 데모",
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
        background: linear-gradient(135deg, #3498db, #2980b9);
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 12px;
        font-size: 1.1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-ok {
        background: linear-gradient(135deg, #d5f5e3, #a9dfbf);
        border-left: 5px solid #2ecc71;
    }
    .result-ng {
        background: linear-gradient(135deg, #fdedec, #f5b7b1);
        border-left: 5px solid #e74c3c;
    }
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        margin-right: 0.5rem;
    }
    .badge-mat { background: #2c3e50; }
    .badge-contam { background: #e67e22; }
    .badge-clean { background: #2980b9; }
</style>
""", unsafe_allow_html=True)

# ─── 모델 로드 (캐싱) ─────────────────────────────────────
@st.cache_resource
def load_everything():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    yolo_model, mobilenet_model = load_models(config, device)
    return yolo_model, mobilenet_model, config, device

# ─── 메인 UI ─────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="main-header">
        <h1>♻️ 2-Stage 통함 파이프라인 실시간 데모</h1>
        <p>1단계: YOLOv8 객체 탐지 ➔ 2단계: MobileNetV3 듀얼헤드(재질/오염) 판별</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("파이프라인 로딩 중... (YOLOv8 + Dual-Head)"):
        yolo_model, mobilenet_model, config, device = load_everything()

    with st.sidebar:
        st.header("⚙️ 설정")
        confidence_threshold = st.slider("재질 판독 신뢰도 컷 (%)", 0, 100, 50, 5) // 100
        st.info(f"🖥️ 연산장치: `{'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}`")

    tab_webcam, tab_upload = st.tabs(["📹 웹캠 실시간", "🖼️ 이미지 업로드"])

    # ── 웹캠 탭 ──
    with tab_webcam:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("실시간 뷰 (Bbox 표출)")
            run = st.toggle("🔴 웹캠 시작", value=False)
            frame_placeholder = st.empty()

        with col2:
            st.subheader("탐지된 객체 목록")
            result_placeholder = st.empty()

        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ 웹캠을 열 수 없습니다.")
            else:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ 프레임 확보 대기 중...")
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    
                    # 2-Stage 파이프라인 가동! (사양: FE_BE_연동스펙.md)
                    api_result = predict(yolo_model, mobilenet_model, pil_img, config, device)
                    data = api_result["data"]
                    
                    html_blocks = []
                    
                    if data["detected"]:
                        for idx, pred in enumerate(data["predictions"]):
                            box = pred["bbox"]
                            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                            
                            conf = pred["confidence"]
                            is_recyc = pred["is_recyclable"]
                            icon = CLASS_ICON.get(pred["label"], "📦")
                            
                            # 색상 지정
                            box_color = (0, 200, 80) if is_recyc else (220, 50, 50)
                            css_class = "result-ok" if is_recyc else "result-ng"
                            contam_css = "badge-contam" if pred["contamination_status"] == "contaminated" else "badge-clean"
                            
                            # 이미지 위에 bbox 그리기
                            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), box_color, 3)
                            label_str = f"#{idx+1} {pred['korean_label']} {conf*100:.0f}%"
                            cv2.putText(frame_rgb, label_str, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                            
                            # 우측 패널에 카드 추가
                            html_blocks.append(f'''
                            <div class="result-box {css_class}">
                                <div style="font-weight:bold; font-size:1.2rem; margin-bottom:0.5rem">
                                    객체 #{idx+1} {icon}
                                </div>
                                <div><span class="badge badge-mat">재질: {pred['korean_label']} ({conf*100:.1f}%)</span></div>
                                <div style="margin-top:0.3rem">
                                    <span class="badge {contam_css}">오염상태: {pred['contamination_status_ko']}</span>
                                </div>
                                <div style="margin-top:0.5rem; font-weight:bold; font-size:1.1rem">
                                    {'✅ 최종: 분리수거함 직행 가능' if is_recyc else '❌ 최종: 재활용 불가 (종량제)'}
                                </div>
                            </div>
                            ''')
                    
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    if html_blocks:
                        result_placeholder.markdown("".join(html_blocks), unsafe_allow_html=True)
                    else:
                        result_placeholder.info("👀 탐지된 재활용품이 없습니다.")

                    # 서버 과부하 방지를 위한 슬립
                    time.sleep(0.05)

                cap.release()

    # ── 이미지 업로드 탭 ──
    with tab_upload:
        uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded)
            col1, col2 = st.columns([1, 1])
            
            with col2:
                with st.spinner("파이프라인 추론 중..."):
                    api_result = predict(yolo_model, mobilenet_model, img, config, device)
            
            with col1:
                # bounding box 시각화
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                
                fig, ax = plt.subplots(1)
                ax.imshow(img)
                ax.axis('off')
                
                data = api_result["data"]
                html_blocks = []
                
                if data["detected"]:
                    for idx, pred in enumerate(data["predictions"]):
                        b = pred["bbox"]
                        is_recyc = pred["is_recyclable"]
                        color = 'green' if is_recyc else 'red'
                        
                        rect = patches.Rectangle((b["x1"], b["y1"]), b["x2"]-b["x1"], b["y2"]-b["y1"], linewidth=3, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)
                        
                        ax.text(b["x1"], max(b["y1"]-10, 0), f"#{idx+1} {pred['korean_label']}", color='white', fontsize=12, bbox=dict(facecolor=color, alpha=0.8, pad=2))
                        
                        css_class = "result-ok" if is_recyc else "result-ng"
                        contam_css = "badge-contam" if pred["contamination_status"] == "contaminated" else "badge-clean"
                        icon = CLASS_ICON.get(pred["label"], "📦")
                        
                        html_blocks.append(f'''
                        <div class="result-box {css_class}">
                            <div style="font-weight:bold; font-size:1.2rem; margin-bottom:0.5rem">
                                객체 #{idx+1} {icon}
                            </div>
                            <div><span class="badge badge-mat">재질: {pred['korean_label']} ({pred['confidence']*100:.1f}%)</span></div>
                            <div style="margin-top:0.3rem">
                                <span class="badge {contam_css}">상태: {pred['contamination_status_ko']}</span>
                            </div>
                            <div style="margin-top:0.5rem; font-weight:bold; font-size:1.1rem">
                                {'✅ 재활용 가능' if is_recyc else '❌ 재활용 불가'}
                            </div>
                        </div>
                        ''')
                
                st.pyplot(fig)
            
            with col2:
                if html_blocks:
                    st.markdown("".join(html_blocks), unsafe_allow_html=True)
                else:
                    st.info("👀 이미지에서 객체를 탐지하지 못했습니다.")


if __name__ == "__main__":
    main()
