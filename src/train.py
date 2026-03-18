"""
전이학습 학습 스크립트 (train.py)
EfficientNet-B0 기반 분리수거 재질 분류 모델 학습
"""

import os
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# ─── 클래스 설정 ─────────────────────────────────────────
CLASS_NAMES = ["plastic", "can", "paper", "glass", "trash"]
CLASS_KO = {
    "plastic": "플라스틱",
    "can": "캔",
    "paper": "종이",
    "glass": "유리",
    "trash": "기타(재활용불가)"
}
RECYCLABLE = {"plastic": True, "can": True, "paper": True, "glass": True, "trash": False}


def build_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
    """EfficientNet-B0 모델 구성 (timm 라이브러리)"""
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes
    )

    if freeze_backbone:
        # 백본 프리징 → 분류 헤드만 학습 (전이학습 1단계)
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("  백본 동결(freeze) 완료 - 분류 헤드만 학습")
    else:
        print("  전체 파라미터 학습 (fine-tuning)")

    return model


def get_transforms(img_size: int = 224):
    """학습/검증용 이미지 변환"""
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  재활용 분류 모델 전이학습")
    print(f"{'='*60}")
    print(f"  디바이스    : {device}")
    print(f"  에폭        : {args.epochs}")
    print(f"  배치 크기   : {args.batch_size}")
    print(f"  학습률      : {args.lr}")
    print(f"  데이터 경로 : {args.data_dir}")
    print(f"{'='*60}\n")

    # ── 데이터 로드 ──
    train_tf, val_tf = get_transforms()
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.exists(train_dir):
        print(f"❌ 오류: 학습 데이터 폴더를 찾을 수 없습니다: {train_dir}")
        print("   data/train/ 폴더 안에 클래스별 이미지를 넣어주세요.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf) if os.path.exists(val_dir) else None

    print(f"  학습 샘플 수: {len(train_dataset)}")
    print(f"  클래스 목록 : {train_dataset.classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    ) if val_dataset else None

    # ── 모델 구성 ──
    num_classes = len(train_dataset.classes)
    print(f"\n[모델 초기화] EfficientNet-B0 (num_classes={num_classes})")
    model = build_model(num_classes=num_classes, pretrained=True, freeze_backbone=args.freeze)
    model = model.to(device)

    # ── 학습 설정 ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 학습 루프 ──
    best_val_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        elapsed = time.time() - start

        log = f"  에폭 [{epoch:2d}/{args.epochs}] | 학습 손실: {train_loss:.4f} | 학습 정확도: {train_acc:.1f}% | 시간: {elapsed:.1f}s"

        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            log += f" | 검증 손실: {val_loss:.4f} | 검증 정확도: {val_acc:.1f}%"

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(args.save_dir, "best_model.pth")
                torch.save(model.state_dict(), save_path)
                log += " ← 베스트 저장 ✅"
        else:
            # 검증셋 없을 경우 매 에폭 저장
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)

        print(log)
        scheduler.step()

    # 클래스 정보 저장
    class_to_idx = train_dataset.class_to_idx
    config = {
        "model_name": "efficientnet_b0",
        "num_classes": num_classes,
        "class_names": list(class_to_idx.keys()),
        "class_to_idx": class_to_idx,
        "class_ko": CLASS_KO,
        "recyclable": RECYCLABLE,
        "input_size": [3, 224, 224],
        "best_val_acc": best_val_acc,
    }
    config_path = os.path.join(args.save_dir, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  학습 완료!")
    print(f"  최고 검증 정확도: {best_val_acc:.1f}%")
    print(f"  모델 저장 위치: {os.path.join(args.save_dir, 'best_model.pth')}")
    print(f"  다음 단계: python src/inference.py  또는  streamlit run src/app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="재활용 분류 전이학습")
    parser.add_argument("--data_dir", type=str, default="data", help="데이터 루트 경로")
    parser.add_argument("--save_dir", type=str, default="models", help="모델 저장 경로")
    parser.add_argument("--epochs", type=int, default=10, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    parser.add_argument("--freeze", action="store_true", default=True,
                        help="백본 동결 여부 (기본: True)")
    parser.add_argument("--no_freeze", dest="freeze", action="store_false",
                        help="백본 동결 해제 (full fine-tuning)")
    args = parser.parse_args()
    train(args)
