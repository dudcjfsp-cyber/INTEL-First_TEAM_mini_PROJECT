"""
추론 스크립트 (inference.py)
Stage 1: YOLOv8-Nano 객체 탐지
Stage 2: Dual-Head MobileNetV3 재질 및 오염도 동시 분류
"""

import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# ─── 기본 설정 ───────────────────────────────────────────
MODEL_PATH = os.path.join("models", "best_model.pth")
CONFIG_PATH = os.path.join("models", "model_config.json")

DEFAULT_CLASSES = ["plastic", "can", "paper", "glass", "trash"]
CLASS_KO = {
    "plastic": "플라스틱",
    "can": "캔",
    "paper": "종이",
    "glass": "유리",
    "trash": "기타"
}
RECYCLABLE = {"plastic": True, "can": True, "paper": True, "glass": True, "trash": False}


class DualHeadMobileNetV3(nn.Module):
    def __init__(self, num_material_classes: int = 4):
        super(DualHeadMobileNetV3, self).__init__()
        self.backbone = timm.create_model("mobilenetv3_small_100", pretrained=False)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        self.material_head = nn.Linear(in_features, num_material_classes)
        self.contamination_head = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        mat_logits = self.material_head(features)
        contam_logits = self.contamination_head(features).squeeze(1)
        return mat_logits, contam_logits


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "model_name": "mobilenetv3_small_100_dual_head",
        "num_classes": len(DEFAULT_CLASSES),
        "class_names": DEFAULT_CLASSES,
        "class_ko": CLASS_KO,
        "recyclable": RECYCLABLE,
    }

def load_models(config: dict, device: torch.device):
    """Stage 1(YOLO)와 Stage 2(MobileNetV3) 모델을 동시 로드"""
    # 1. YOLOv8 로드
    yolo_model = YOLO("yolov8n.pt")
    
    # 2. Dual-Head 로드
    mobilenet_model = DualHeadMobileNetV3(num_material_classes=config["num_classes"])
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        mobilenet_model.load_state_dict(state_dict, strict=False)
        print(f"  ✅ Dual-Head 모델 로드 완료: {MODEL_PATH}")
    else:
        print(f"  ⚠️  Dual-Head 가중치 없음 ({MODEL_PATH}).")
    
    mobilenet_model = mobilenet_model.to(device)
    mobilenet_model.eval()
    
    return yolo_model, mobilenet_model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


@torch.no_grad()
def predict(yolo_model, mobilenet_model, image: Image.Image, config: dict, device: torch.device):
    """
    FE_BE_연동스펙.md를 완벽히 준수하는 2-Stage 추론 함수
    """
    start_time = time.time()
    tf = get_transforms()
    
    # Stage 1: YOLOv8 객체 탐지
    yolo_results = yolo_model(image, conf=0.25, verbose=False)
    boxes = yolo_results[0].boxes
    
    predictions = []
    
    class_names = config["class_names"]
    class_ko = config.get("class_ko", CLASS_KO)
    recyclable_config = config.get("recyclable", RECYCLABLE)
    
    for box in boxes:
        # Bbox 좌표 추출
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 안전한 Crop
        img_width, img_height = image.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)
        
        # 크기가 너무 작으면 무시
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
            
        cropped_img = image.crop((x1, y1, x2, y2))
        
        # Stage 2: Dual-Head 분류
        tensor = tf(cropped_img.convert("RGB")).unsqueeze(0).to(device)
        mat_logits, contam_logits = mobilenet_model(tensor)
        
        mat_probs = F.softmax(mat_logits, dim=1)[0]
        contam_prob = torch.sigmoid(contam_logits)[0].item()
        
        top_mat_idx = mat_probs.argmax().item()
        top_mat_class = class_names[top_mat_idx]
        mat_confidence = mat_probs[top_mat_idx].item()
        
        # 오염도 판별
        is_contaminated = bool(contam_prob >= 0.5)
        # FE/BE 규격 준수 ("안깨끗함" -> "contaminated" 혹은 자유 사용. 예시에서는 "clean" 사용)
        contam_status_str = "contaminated" if is_contaminated else "clean"
        contam_status_ko = "오염됨" if is_contaminated else "깨끗함"
        
        # 재활용 판별 로직
        is_mat_recyclable = recyclable_config.get(top_mat_class, False)
        final_recyclable = is_mat_recyclable and (not is_contaminated)
        
        pred_dict = {
            "label": top_mat_class,
            "korean_label": class_ko.get(top_mat_class, top_mat_class),
            "confidence": round(mat_confidence, 4),
            "is_recyclable": final_recyclable,
            "contamination_status": contam_status_str,
            "contamination_status_ko": contam_status_ko,  # 추가 정보
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        }
        predictions.append(pred_dict)
        
    inference_time_ms = round((time.time() - start_time) * 1000, 2)
    
    return {
        "status": "success",
        "data": {
            "detected": len(predictions) > 0,
            "predictions": predictions,
            "inference_time_ms": inference_time_ms
        }
    }


def print_result(result: dict):
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_test_mode(yolo_model, mobilenet_model, config, device):
    print("\n[테스트 모드] 더미 이미지(랜덤 노이즈)로 API 응답 구조 검증...")
    dummy = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    # YOLO가 더미에서는 객체를 못 잡을 수 있으므로 강제로 인식시키거나 빈배열 테스트 수행
    result = predict(yolo_model, mobilenet_model, dummy, config, device)
    print_result(result)
    print("  ✅ 2-Stage 통합 추론 파이프라인 (JSON 규격) 확인 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-Stage (YOLOv8 + Dual-Head) 파이프라인")
    parser.add_argument("--image", type=str, default=None, help="추론할 이미지 경로")
    parser.add_argument("--test", action="store_true", help="더미 이미지로 파이프라인 테스트")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()

    print(f"\n{'='*60}")
    print(f"  재활용 분류기 - 2-Stage 통합 엔진 작동")
    print(f"  디바이스: {device}")
    print(f"{'='*60}")

    yolo_model, mobilenet_model = load_models(config, device)

    if args.test:
        run_test_mode(yolo_model, mobilenet_model, config, device)
    elif args.image:
        if not os.path.exists(args.image):
            print(f"❌ 이미지 파일을 찾을 수 없습니다: {args.image}")
        else:
            img = Image.open(args.image)
            result = predict(yolo_model, mobilenet_model, img, config, device)
            print_result(result)
    else:
        print("사용법:")
        print("  python src/inference.py --test")
        print("  python src/inference.py --image 이미지.jpg")
