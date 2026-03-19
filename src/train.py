"""
전이학습 학습 스크립트 (train.py)
MobileNetV3-Small 기반 듀얼 헤드 (재질, 오염도) 동시 판별 모델 학습
"""

import os
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

# 새로 만든 데이터셋 임포트
from dataset import RecyclingDualHeadDataset

# ─── 클래스 설정 ─────────────────────────────────────────
CLASS_NAMES = ["plastic", "can", "paper", "glass", "trash"] # 5개 클래스로 확장
CLASS_KO = {
    "plastic": "플라스틱",
    "can": "캔",
    "paper": "종이",
    "glass": "유리",
    "trash": "기타(쓰레기)"
}
# 오염 헤드 판별 명칭
CONTAM_KO = {0: "깨끗함", 1: "오염됨"}

# 재질별 기본 재활용 가능 여부 (trash는 False)
RECYCLABLE = {
    "plastic": True, 
    "can": True, 
    "paper": True, 
    "glass": True,
    "trash": False
}


class DualHeadMobileNetV3(nn.Module):
    """MobileNetV3 Small을 백본으로 하는 듀얼 헤드 아키텍처"""
    def __init__(self, num_material_classes: int = 5, pretrained: bool = True, freeze_backbone: bool = True):
        super(DualHeadMobileNetV3, self).__init__()
        
        # PRD/TRD에 명시된 MobileNetV3 Small 로드
        self.backbone = timm.create_model("mobilenetv3_small_100", pretrained=pretrained)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("  백본 동결(freeze) 완료 - 분류 헤드만 학습")
            
        # 백본의 마지막 피처 차원 파악 (MobileNetV3 본래 classifier의 in_features)
        in_features = self.backbone.classifier.in_features
        
        # 기존 classifier(단일 헤드) 제거
        self.backbone.classifier = nn.Identity()
        
        # 새로운 듀얼 헤드 부착
        # 1. 재질 분류 헤드 (4개 클래스)
        self.material_head = nn.Linear(in_features, num_material_classes)
        
        # 2. 오염도 판별 헤드 (이진 분류용 1개 클래스 로그 로짓)
        self.contamination_head = nn.Linear(in_features, 1)

    def forward(self, x):
        # [B, in_features]
        features = self.backbone(x)
        
        mat_logits = self.material_head(features) # [B, 4]
        contam_logits = self.contamination_head(features).squeeze(1) # [B]
        
        return mat_logits, contam_logits


def train_one_epoch(model, loader, criterion_mat, criterion_contam, optimizer, device):
    model.train()
    total_loss = 0.0
    correct_mat, correct_contam, total = 0, 0, 0

    for images, (mat_labels, contam_labels) in loader:
        images = images.to(device)
        mat_labels = mat_labels.to(device)
        # Binary Cross Entropy는 float를 원함
        contam_labels = contam_labels.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        mat_logits, contam_logits = model(images)
        
        # Loss 계산
        loss_mat = criterion_mat(mat_logits, mat_labels)
        loss_contam = criterion_contam(contam_logits, contam_labels)
        loss = loss_mat + loss_contam
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        
        # 정확도 기록용 (Material)
        _, mat_pred = mat_logits.max(1)
        correct_mat += mat_pred.eq(mat_labels).sum().item()
        
        # 정확도 기록용 (Contamination: 0.0 이상이면 1(오염됨), 미만이면 0(깨끗함))
        contam_pred = (contam_logits > 0.0).float()
        correct_contam += contam_pred.eq(contam_labels).sum().item()
        
        total += images.size(0)

    return total_loss / total, 100.0 * correct_mat / total, 100.0 * correct_contam / total


@torch.no_grad()
def validate(model, loader, criterion_mat, criterion_contam, device):
    model.eval()
    total_loss = 0.0
    correct_mat, correct_contam, total = 0, 0, 0

    for images, (mat_labels, contam_labels) in loader:
        images = images.to(device)
        mat_labels = mat_labels.to(device)
        contam_labels = contam_labels.to(device, dtype=torch.float32)

        mat_logits, contam_logits = model(images)
        
        loss_mat = criterion_mat(mat_logits, mat_labels)
        loss_contam = criterion_contam(contam_logits, contam_labels)
        loss = loss_mat + loss_contam

        total_loss += loss.item() * images.size(0)
        
        _, mat_pred = mat_logits.max(1)
        correct_mat += mat_pred.eq(mat_labels).sum().item()
        
        contam_pred = (contam_logits > 0.0).float()
        correct_contam += contam_pred.eq(contam_labels).sum().item()
        
        total += images.size(0)

    return total_loss / total, 100.0 * correct_mat / total, 100.0 * correct_contam / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  듀얼 헤드 모델 전이학습 (재질 + 오염도)")
    print(f"{'='*60}")
    print(f"  디바이스    : {device}")
    print(f"  에폭        : {args.epochs}")
    print(f"  배치 크기   : {args.batch_size}")
    print(f"  데이터 경로 : {args.data_dir}")
    print(f"{'='*60}\n")

    # ── 데이터 로드 (커스텀 듀얼 헤드 데이터셋 활용) ──
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    train_dir = os.path.join(args.data_dir, "train")
    if not os.path.exists(train_dir):
        print(f"❌ 오류: 학습 데이터 폴더를 찾을 수 없습니다: {train_dir}")
        return

    train_dataset = RecyclingDualHeadDataset(train_dir, class_to_idx, is_train=True)
    
    print(f"  학습 샘플 수: {len(train_dataset)}")
    print(f"  재질 클래스 : {CLASS_NAMES}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
    )
    
    # Val 데이터셋이 있을 경우 로드
    val_dir = os.path.join(args.data_dir, "val")
    val_loader = None
    if os.path.exists(val_dir):
        val_dataset = RecyclingDualHeadDataset(val_dir, class_to_idx, is_train=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ── 모델 구성 ──
    num_classes = len(CLASS_NAMES)
    print(f"\n[모델 초기화] Dual-Head MobileNetV3 Small")
    model = DualHeadMobileNetV3(num_material_classes=num_classes, pretrained=True, freeze_backbone=args.freeze)
    
    # 기존 가중치 이어서 학습 (파인튜닝용)
    if args.resume:
        weight_path = os.path.join(args.save_dir, "best_model.pth")
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
            print(f"  ✅ 기존 가중치 로드 완료 (파인튜닝 모드): {weight_path}")
        else:
            print(f"  ⚠️ 기존 가중치 없음. ImageNet 사전학습 가중치로 시작합니다.")
    
    model = model.to(device)

    # ── 학습 설정 ──
    criterion_mat = nn.CrossEntropyLoss()
    criterion_contam = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 학습 루프 ──
    best_mat_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, tr_mat_acc, tr_contam_acc = train_one_epoch(
            model, train_loader, criterion_mat, criterion_contam, optimizer, device
        )
        elapsed = time.time() - start

        log = f"에폭 [{epoch:2d}/{args.epochs}] 손실: {train_loss:.4f} | 재질: {tr_mat_acc:.1f}% 오염: {tr_contam_acc:.1f}% | 시간: {elapsed:.1f}s"

        if val_loader:
            val_loss, va_mat_acc, va_contam_acc = validate(
                model, val_loader, criterion_mat, criterion_contam, device
            )
            log += f" || [검증] 재질: {va_mat_acc:.1f}% 오염: {va_contam_acc:.1f}%"

            # 베스트 모델 기준을 재질 정확도로 둠
            if va_mat_acc > best_mat_acc:
                best_mat_acc = va_mat_acc
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
                log += " ← 베스트 저장 ✅"
        else:
            # 검증셋 없을 경우 매 에폭 저장
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

        print(log)
        scheduler.step()

    # 클래스 정보 저장
    config = {
        "model_name": "mobilenetv3_small_100_dual_head",
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "class_to_idx": class_to_idx,
        "class_ko": CLASS_KO,
        "recyclable": RECYCLABLE,
        "input_size": [3, 224, 224],
    }
    with open(os.path.join(args.save_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  학습 완료 (모델 저장: {args.save_dir}/best_model.pth)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="듀얼 헤드 재활용 모델 전이학습")
    parser.add_argument("--data_dir", type=str, default="data", help="데이터 경로")
    parser.add_argument("--save_dir", type=str, default="models", help="모델 저장 경로")
    parser.add_argument("--epochs", type=int, default=10, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    parser.add_argument("--freeze", action="store_true", default=True)
    parser.add_argument("--no_freeze", dest="freeze", action="store_false")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="기존 best_model.pth를 불러와서 이어서 학습 (파인튜닝용)")
    args = parser.parse_args()
    train(args)
