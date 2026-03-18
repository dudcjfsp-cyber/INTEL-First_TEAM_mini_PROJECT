# 🛠️ Return To Me - 기술 요구사항 정의서 (TRD)

## 1. 시스템 아키텍처 (System Architecture)
본 시스템은 실시간 영상 스트림을 처리하기 위해 **Edge-first 추론 구조**를 채택하며, 고성능 탐지와 효율적인 분류를 결합한 2단계 파이프라인으로 구성됩니다.

### 1-1. 2단계 추론 파이프라인 (2-Stage Inference Pipeline)
1.  **Stage 1: Object Detection (YOLOv8-Nano)**
    - 입력 프레임에서 재활용품(쓰레기) 객체의 위치를 탐지합니다.
    - 다수 객체가 존재할 경우, Conf Score가 가장 높은 객체 또는 모든 객체에 대해 Bounding Box를 생성합니다.
2.  **Stage 2: Feature Classification (MobileNetV3 Small)**
    - Stage 1에서 추출된 Bounding Box 영역을 Crop 하여 입력으로 사용합니다.
    - 해당 객체의 세부 재질(유리/종이/플라스틱/캔) 및 오염 상태를 최종 분류합니다.

## 2. 모델 명세 (Model Specifications)

| 구분 | 모델명 | 프레임워크 | 역할 | 비고 |
| --- | --- | --- | --- | --- |
| **Detection** | YOLOv8-Nano | PyTorch / Ultralytics | 배경 대비 객체 위치 추출 | 가볍고 빠른 실시간 탐지 최적화 |
| **Classification** | MobileNetV3 Small | PyTorch / timm | 재질 분류 및 오염도 판별 | 소형 기기 이식에 최적화된 구조 |

### 2-1. 데이터 전처리 (Preprocessing)
- **Resize**: YOLO(640x640), MobileNetV3(224x224)
- **Normalization**: ImageNet stats (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225])
- **Format Conversion**: OpenCV(BGR) -> PIL/PyTorch(RGB)

## 3. 소프트웨어 요구사항 (Software Requirements)

### 3-1. 개발 환경
- **Language**: Python 3.10.x 이상
- **딥러닝**: PyTorch 2.1.0+, Torchvision 0.16.0+
- **추론 최적화**: ONNX Runtime (CPU/GPU 가속 가능 환경)
- **라이브러리**: `opencv-python`, `ultralytics`, `timm`, `numpy`, `streamlit`

### 3-2. 외부 연동 인터페이스
- **Webcam API**: OpenCV `VideoCapture`를 통한 프레임 획득 (최소 720p 30fps 지원 권장)
- **UI Interaction**: Streamlit 기반의 실시간 프레임 렌더링 및 결과 오버레이
- **Stitch/Antigravity**: UI 디자인 에셋 반영 및 프론트엔드 코드 동기화

## 4. 성능 요구사항 (Performance Requirements)

- **Latency (지연 시간)**: 프레임당 전체 추론 시간(Detection + Classification) 100ms 이내 목표 (CPU 기준).
- **Throughput (처리량)**: 최소 10~15 FPS 이상의 실시간 화면 표시 보장.
- **Accuracy (정확도)**: PoC 단계 기준 주요 4종 클래스에 대해 Top-1 Accuracy 85% 이상 확보.
- **Confidence Threshold**: 0.5 미만의 예측 결과는 "미인식" 또는 "기타"로 처리하여 오동작 방지.

## 5. 예외 처리 (Error Handling)
- **No Object Detected**: 객체가 탐지되지 않을 경우 "객체를 비춰주세요" 안내 문구 표시.
- **Low Confidence**: 신뢰도가 낮을 경우 "더 가까이 비춰주세요" 또는 "재시도" 알림.
- **Camera Error**: 카메라 연결 해제 시 에러 메시지 출력 및 재연결 시도.

## 6. 테스트 및 검증 방향
- **Unit Test**: 개별 모델(YOLO, MobileNetV3)의 단일 이미지 추론 결과 검증.
- **Integration Test**: 2단계 파이프라인 연결 시의 지연 시간 측정 및 데이터 전달 무결성 확인.
- **Scenario Test**: 실제 재활용품(구겨진 캔, 라벨 붙은 페트병 등)을 활용한 현장 시연 테스트.

---
*Created by Team "Return To Me" (권경일, 유지현, 권순현)*
