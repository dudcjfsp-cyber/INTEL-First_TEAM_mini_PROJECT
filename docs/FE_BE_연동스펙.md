# 🤝 Frontend - Backend 연동 규격서 (API Spec)

## 1. 개요
본 문서는 **Return To Me** 프로젝트의 React 프론트엔드와 FastAPI 백엔드 간의 통신 규격을 정의합니다. 실시간 영상 분석을 위해 HTTP 기반의 RESTful API를 사용합니다.

## 2. 기본 정보
- **Base URL**: `http://localhost:8000` (개발 환경 기준)
- **통신 방식**: HTTP POST
- **데이터 포맷**: JSON (Response), Multipart/Form-data (Request - Image)

## 3. API 상세 내역

### 3-1. 실시간 영상 분석 (Real-time Analysis)
카메라 앱에서 캡처된 프레임을 서버로 전송하여 객체 탐지 및 분류 결과를 요청합니다.

- **Endpoint**: `/api/v1/analyze`
- **Method**: `POST`
- **Request Headers**:
  - `Content-Type`: `multipart/form-data`
- **Request Body**:

| 필드명 | 타입 | 필수 여부 | 설명 |
| --- | --- | --- | --- |
| `file` | Binary (Image) | 필수 | 전송할 이미지 파일 (JPEG 또는 PNG 권장) |

- **Response Body (Success)**:

```json
{
  "status": "success",
  "data": {
    "detected": true,
    "predictions": [
      {
        "label": "plastic_bottle",
        "korean_label": "플라스틱 병",
        "confidence": 0.98,
        "is_recyclable": true,
        "contamination_status": "clean",
        "bbox": {
          "x1": 100,
          "y1": 50,
          "x2": 300,
          "y2": 450
        }
      }
    ],
    "inference_time_ms": 75.2
  }
}
```

- **Response Body (No Object Detected)**:

```json
{
  "status": "success",
  "data": {
    "detected": false,
    "predictions": [],
    "inference_time_ms": 20.5
  }
}
```

## 4. 연동 로직 및 흐름 제어 (Flow Control)

1.  **프레임 캡처**: 프론트엔드(React)는 1초(1 FPS) 간격으로 카메라 영상을 캡처합니다.
2.  **전송 제어 (Wait & Skip)**:
    - 현재 프레임을 서버로 전송한 후, 응답(`Response`)이 올 때까지 다음 프레임 전송을 대기(Skip)합니다.
    - 응답을 받은 후에만 다음 1초 주기의 프레임을 전송할 수 있습니다.
3.  **결과 시각화**:
    - 서버에서 받은 `bbox` 정보를 바탕으로 캔버스(Canvas) 위에 바운딩 박스를 그립니다.
    - `korean_label`과 `is_recyclable` 결과를 우상단 또는 박스 근처에 표시합니다.

## 5. 에러 처리 (Error Codes)

| 상태 코드 | 설명 | 프론트엔드 대응 |
| --- | --- | --- |
| `400 Bad Request` | 이미지 파일 누락 또는 형식이 잘못됨 | 사용자에게 "올바른 이미지가 아닙니다" 메시지 출력 |
| `413 Payload Too Large` | 이미지 용량이 너무 큼 | 이미지 리사이징(640px 이하) 후 재전송 |
| `429 Too Many Requests` | 분석 요청이 너무 빈번함 | 전송 간격 재조정 (1 FPS 유지 확인) |
| `500 Internal Server Error` | 서버 내부 로직(모델 추론 등) 에러 | "서버 오류" 표시 후 3초 뒤 재시도 |

---
*Created by Team "Return To Me"*
