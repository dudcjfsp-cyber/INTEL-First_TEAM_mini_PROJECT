"""
캐글 Garbage Classification 데이터셋을 프로젝트 구조에 맞게 분류/복사하는 스크립트
cardboard+paper → paper, metal → can 으로 매핑
80/20 비율로 train/val 분리
"""

import os
import shutil
import random

random.seed(42)

# ─── 경로 설정
SRC_BASE = os.path.join("Garbage classification", "Garbage classification")
DST_TRAIN = os.path.join("data", "train")
DST_VAL = os.path.join("data", "val")

# ─── 매핑: 캐글 폴더명 → 프로젝트 클래스명
MAPPING = {
    "cardboard": "paper",    # 골판지 → 종이류
    "glass":     "glass",    # 유리
    "metal":     "can",      # 금속 → 캔
    "paper":     "paper",    # 종이
    "plastic":   "plastic",  # 플라스틱
    "trash":     "trash",    # 기타 (재활용 불가)
}

VAL_RATIO = 0.2  # 검증 세트 비율

def main():
    print("=" * 60)
    print("  캐글 데이터셋 → 프로젝트 구조 변환")
    print("=" * 60)

    # 대상 폴더 생성
    for cls in set(MAPPING.values()):
        os.makedirs(os.path.join(DST_TRAIN, cls), exist_ok=True)
        os.makedirs(os.path.join(DST_VAL, cls), exist_ok=True)

    stats = {}
    total_copied = 0

    for src_folder, dst_class in MAPPING.items():
        src_dir = os.path.join(SRC_BASE, src_folder)
        if not os.path.exists(src_dir):
            print(f"  ⚠️ 폴더 없음: {src_dir}")
            continue

        # 이미지 파일 목록
        files = [f for f in os.listdir(src_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        random.shuffle(files)

        split_idx = int(len(files) * (1 - VAL_RATIO))
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        # 파일명 충돌 방지: 원본 폴더명을 접두사로
        for f in train_files:
            src_path = os.path.join(src_dir, f)
            dst_name = f"{src_folder}_{f}" if src_folder != dst_class else f
            dst_path = os.path.join(DST_TRAIN, dst_class, dst_name)
            shutil.copy2(src_path, dst_path)

        for f in val_files:
            src_path = os.path.join(src_dir, f)
            dst_name = f"{src_folder}_{f}" if src_folder != dst_class else f
            dst_path = os.path.join(DST_VAL, dst_class, dst_name)
            shutil.copy2(src_path, dst_path)

        n_train = len(train_files)
        n_val = len(val_files)
        total_copied += n_train + n_val

        if dst_class not in stats:
            stats[dst_class] = {"train": 0, "val": 0, "sources": []}
        stats[dst_class]["train"] += n_train
        stats[dst_class]["val"] += n_val
        stats[dst_class]["sources"].append(f"{src_folder}({n_train+n_val})")

        print(f"  [{src_folder}] → [{dst_class}]  train: {n_train}장, val: {n_val}장")

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"  변환 완료! 총 {total_copied}장 복사됨")
    print(f"{'='*60}")
    print(f"\n  {'클래스':<12} {'Train':>8} {'Val':>8} {'합계':>8}  출처")
    print(f"  {'─'*60}")

    CLASS_KO = {"plastic": "플라스틱", "can": "캔", "paper": "종이", "glass": "유리", "trash": "기타"}
    for cls in ["plastic", "can", "paper", "glass", "trash"]:
        if cls in stats:
            s = stats[cls]
            total = s["train"] + s["val"]
            sources = " + ".join(s["sources"])
            print(f"  {CLASS_KO[cls]:<10} {s['train']:>8}장 {s['val']:>8}장 {total:>8}장  ← {sources}")

    print(f"\n  다음 단계: python src/train.py --epochs 10")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
