"""
추론 스크립트 (inference.py)
학습된 EfficientNet-B0 모델로 이미지 분류 수행
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms


# ─── 기본 설정 ───────────────────────────────────────────
MODEL_PATH = os.path.join("models", "best_model.pth")
CONFIG_PATH = os.path.join("models", "model_config.json")

# 기본 클래스 정보 (config 없을 때 fallback)
DEFAULT_CLASSES = ["plastic", "can", "paper", "glass", "trash"]
CLASS_KO = {
    "plastic": "플라스틱",
    "can": "캔",
    "paper": "종이",
    "glass": "유리",
    "trash": "기타(재활용불가)"
}
RECYCLABLE = {"plastic": True, "can": True, "paper": True, "glass": True, "trash": False}


def load_config():
    """저장된 모델 설정 파일 로드"""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "model_name": "efficientnet_b0",
        "num_classes": len(DEFAULT_CLASSES),
        "class_names": DEFAULT_CLASSES,
        "class_ko": CLASS_KO,
        "recyclable": RECYCLABLE,
    }


def load_model(config: dict, device: torch.device):
    """학습된 모델 로드"""
    model = timm.create_model(
        config["model_name"],
        pretrained=False,
        num_classes=config["num_classes"]
    )

    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  ✅ 모델 로드 완료: {MODEL_PATH}")
    else:
        print(f"  ⚠️  저장된 모델 없음 ({MODEL_PATH}). 사전학습 가중치로 실행.")
        model = timm.create_model(config["model_name"], pretrained=True, num_classes=config["num_classes"])

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
    """단일 PIL 이미지 추론"""
    tf = get_transforms()
    tensor = tf(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]

    class_names = config["class_names"]
    class_ko = config.get("class_ko", CLASS_KO)
    recyclable = config.get("recyclable", RECYCLABLE)

    top_idx = probs.argmax().item()
    top_class = class_names[top_idx]
    top_conf = probs[top_idx].item() * 100
    is_recyclable = recyclable.get(top_class, False)

    result = {
        "class_en": top_class,
        "class_ko": class_ko.get(top_class, top_class),
        "confidence": top_conf,
        "recyclable": is_recyclable,
        "all_probs": {class_names[i]: probs[i].item() * 100 for i in range(len(class_names))}
    }
    return result


def print_result(result: dict):
    status = "✅ 재활용 가능" if result["recyclable"] else "❌ 재활용 불가"
    print(f"\n{'─'*40}")
    print(f"  예측 결과  : {result['class_ko']} ({result['class_en']})")
    print(f"  재활용 여부: {status}")
    print(f"  신뢰도     : {result['confidence']:.1f}%")
    print(f"{'─'*40}")
    print("  전체 확률 분포:")
    for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob / 5)
        ko = CLASS_KO.get(cls, cls)
        print(f"    {ko:12s} {bar:<20s} {prob:.1f}%")
    print()


def run_test_mode(model, config, device):
    """더미 이미지로 파이프라인 테스트"""
    print("\n[테스트 모드] 더미 이미지(랜덤 노이즈)로 추론 파이프라인 검증...")
    dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    result = predict(model, dummy, config, device)
    print_result(result)
    print("  ✅ 추론 파이프라인 정상 동작 확인!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="재활용 분류 추론")
    parser.add_argument("--image", type=str, default=None, help="추론할 이미지 경로")
    parser.add_argument("--test", action="store_true", help="더미 이미지로 파이프라인 테스트")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()

    print(f"\n{'='*60}")
    print(f"  재활용 분류 추론 모듈")
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
