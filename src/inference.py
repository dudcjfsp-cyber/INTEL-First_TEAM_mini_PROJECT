"""
추론 스크립트 (inference.py)
학습된 Dual-Head MobileNetV3 모델로 재질 및 오염도 동시 분류 수행
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms


# ─── 기본 설정 ───────────────────────────────────────────
MODEL_PATH = os.path.join("models", "best_model.pth")
CONFIG_PATH = os.path.join("models", "model_config.json")

# 기본 클래스 정보 (config 없을 때 fallback)
DEFAULT_CLASSES = ["plastic", "can", "paper", "glass"]
CLASS_KO = {
    "plastic": "플라스틱",
    "can": "캔",
    "paper": "종이",
    "glass": "유리"
}
RECYCLABLE = {"plastic": True, "can": True, "paper": True, "glass": True}


class DualHeadMobileNetV3(nn.Module):
    """train.py에서 정의된 아키텍처와 동일"""
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
    """저장된 모델 설정 파일 로드"""
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


def load_model(config: dict, device: torch.device):
    """학습된 듀얼 헤드 모델 로드"""
    model = DualHeadMobileNetV3(num_material_classes=config["num_classes"])

    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"  ✅ 모델 로드 완료: {MODEL_PATH}")
    else:
        print(f"  ⚠️  저장된 모델 가중치 없음 ({MODEL_PATH}). 초기화된 상태로 실행합니다.")

    model = model.to(device)
    model.eval()
    return model


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


@torch.no_grad()
def predict(model, image: Image.Image, config: dict, device: torch.device):
    """단일 PIL 이미지 추론 (재질 및 오염도 동시 반환)"""
    tf = get_transforms()
    tensor = tf(image.convert("RGB")).unsqueeze(0).to(device)
    
    # 듀얼 추론
    mat_logits, contam_logits = model(tensor)
    
    # 확률 계산
    mat_probs = F.softmax(mat_logits, dim=1)[0]
    # BCEWithLogitsLoss의 짝꿍은 Sigmoid
    contam_prob = torch.sigmoid(contam_logits)[0].item()

    class_names = config["class_names"]
    class_ko = config.get("class_ko", CLASS_KO)
    recyclable_config = config.get("recyclable", RECYCLABLE)

    # 재질 판별
    top_mat_idx = mat_probs.argmax().item()
    top_mat_class = class_names[top_mat_idx]
    top_mat_conf = mat_probs[top_mat_idx].item() * 100
    
    # 오염도 판별 (50% 기준)
    is_contaminated = bool(contam_prob >= 0.5)
    contam_conf = (contam_prob if is_contaminated else (1.0 - contam_prob)) * 100 

    # 최종 재활용 가능 여부 판별 로직:
    # 1. 재질 자체가 재활용이 가능한가? AND 2. 오염되지 않았는가?
    is_mat_recyclable = recyclable_config.get(top_mat_class, False)
    final_recyclable = is_mat_recyclable and (not is_contaminated)

    result = {
        "material": {
            "class_en": top_mat_class,
            "class_ko": class_ko.get(top_mat_class, top_mat_class),
            "confidence": top_mat_conf
        },
        "contamination": {
            "is_contaminated": is_contaminated,
            "status_ko": "오염됨" if is_contaminated else "깨끗함",
            "confidence": contam_conf
        },
        "final_recyclable": final_recyclable,
        "all_probs": {class_names[i]: mat_probs[i].item() * 100 for i in range(len(class_names))}
    }
    return result


def print_result(result: dict):
    mat = result["material"]
    contam = result["contamination"]
    status_icon = "✅ (재활용 가능)" if result["final_recyclable"] else "❌ (재활용 불가)"

    print(f"\n{'─'*45}")
    print(f"  [1] 재질 판별 : {mat['class_ko']:<8s} ({mat['confidence']:.1f}% 신뢰도)")
    print(f"  [2] 오염 상태 : {contam['status_ko']:<8s} ({contam['confidence']:.1f}% 신뢰도)")
    print(f"{'─'*45}")
    print(f"  ▶ 최종 결과   : {status_icon}")
    print(f"{'─'*45}")
    
    print("  [전체 재질 확률 분포]")
    for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob / 5)
        ko = CLASS_KO.get(cls, cls)
        print(f"    {ko:12s} {bar:<20s} {prob:.1f}%")
    print()


def run_test_mode(model, config, device):
    """더미 이미지로 파이프라인 테스트"""
    print("\n[테스트 모드] 더미 이미지(랜덤 노이즈)로 듀얼 헤드 파이프라인 검증...")
    dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    result = predict(model, dummy, config, device)
    print_result(result)
    print("  ✅ 듀얼 헤드 추론 파이프라인 정상 동작 확인!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="재질 및 오염도 동시 추론")
    parser.add_argument("--image", type=str, default=None, help="추론할 이미지 경로")
    parser.add_argument("--test", action="store_true", help="더미 이미지로 파이프라인 테스트")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()

    print(f"\n{'='*60}")
    print(f"  재활용 분류기 - 듀얼 헤드 엔진 작동")
    print(f"  디바이스: {device}")
    print(f"{'='*60}")

    model = load_model(config, device)

    if args.test:
        run_test_mode(model, config, device)
    elif args.image:
        if not os.path.exists(args.image):
            print(f"❌ 이미지 파일을 찾을 수 없습니다: {args.image}")
        else:
            img = Image.open(args.image)
            result = predict(model, img, config, device)
            print_result(result)
    else:
        print("사용법:")
        print("  python src/inference.py --test              # 파이프라인 테스트")
        print("  python src/inference.py --image 이미지.jpg  # 이미지 추론")
