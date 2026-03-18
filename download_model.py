"""
EfficientNet-B0 사전학습 모델 다운로드 스크립트
Hugging Face (timm 라이브러리)에서 ImageNet 사전학습 가중치를 다운로드합니다.
"""

import os
import torch
import timm
import json

# ─── 설정 ───────────────────────────────────────────────
MODEL_NAME = "efficientnet_b0"        # timm 모델 이름
NUM_CLASSES = 5                        # 분류 클래스 수 (plastic, can, paper, glass, trash)
SAVE_DIR = "models"
PRETRAINED_SAVE_PATH = os.path.join(SAVE_DIR, "efficientnet_b0_pretrained.pth")
CONFIG_SAVE_PATH = os.path.join(SAVE_DIR, "model_config.json")

# 클래스 정의
CLASS_NAMES = ["plastic", "can", "paper", "glass", "trash"]
CLASS_KO = {
    "plastic": "플라스틱",
    "can": "캔",
    "paper": "종이",
    "glass": "유리",
    "trash": "기타(재활용불가)"
}
RECYCLABLE = {
    "plastic": True,
    "can": True,
    "paper": True,
    "glass": True,
    "trash": False
}

def download_and_save_model():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("  EfficientNet-B0 사전학습 모델 다운로드")
    print("  출처: Hugging Face (timm 라이브러리)")
    print("=" * 60)

    print(f"\n[1/3] Hugging Face에서 {MODEL_NAME} 가중치 다운로드 중...")
    print("      (첫 실행 시 수백 MB 다운로드될 수 있습니다)\n")

    # pretrained=True → Hugging Face timm을 통해 자동 다운로드
    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=0  # 분류 헤드 제거 (피처 추출기만)
    )
    model.eval()

    print(f"[2/3] 사전학습 가중치를 {PRETRAINED_SAVE_PATH} 에 저장 중...")
    torch.save(model.state_dict(), PRETRAINED_SAVE_PATH)
    print(f"      ✅ 저장 완료: {PRETRAINED_SAVE_PATH}")

    # 모델 설정 정보 저장
    config = {
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "class_ko": CLASS_KO,
        "recyclable": RECYCLABLE,
        "input_size": [3, 224, 224],
        "pretrained": True,
        "pretrained_path": PRETRAINED_SAVE_PATH,
    }
    with open(CONFIG_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[3/3] 모델 설정 저장 완료: {CONFIG_SAVE_PATH}")

    # 모델 구조 요약 출력
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  모델 정보 요약")
    print(f"{'='*60}")
    print(f"  모델명     : {MODEL_NAME}")
    print(f"  파라미터 수: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"  분류 클래스: {NUM_CLASSES}개")
    for i, cls in enumerate(CLASS_NAMES):
        icon = "✅" if RECYCLABLE[cls] else "❌"
        print(f"    [{i}] {CLASS_KO[cls]} {icon}")
    print(f"\n  전이학습 준비 완료! 다음 단계: python src/train.py")
    print(f"{'='*60}\n")

    return model

if __name__ == "__main__":
    download_and_save_model()
