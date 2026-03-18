"""
웹캠 데이터 수집 도구 (capture_data.py)
웹캠으로 사진을 찍어서 학습 데이터 폴더에 자동 저장합니다.

사용법:
    python src/capture_data.py                  # 기본: plastic 클래스
    python src/capture_data.py --label can      # 캔 클래스
    python src/capture_data.py --label glass    # 유리 클래스

조작법:
    [Space] / [Enter]  → 사진 캡처 & 저장
    [1]~[5]            → 클래스 전환 (1:플라스틱, 2:캔, 3:종이, 4:유리, 5:기타)
    [Q] / [ESC]        → 종료
"""

import os
import cv2
import time
import argparse
from datetime import datetime

# ─── 클래스 설정 ─────────────────────────────────────────
CLASSES = {
    "1": "plastic",
    "2": "can",
    "3": "paper",
    "4": "glass",
    "5": "trash",
}
CLASS_KO = {
    "plastic": "플라스틱",
    "can": "캔",
    "paper": "종이",
    "glass": "유리",
    "trash": "기타(재활용불가)"
}
CLASS_COLOR = {
    "plastic": (255, 165, 0),    # 주황
    "can": (100, 100, 255),      # 빨간
    "paper": (200, 200, 100),    # 하늘
    "glass": (100, 255, 100),    # 초록
    "trash": (128, 128, 128),    # 회색
}

DATA_DIR = "data"


def main(initial_label: str):
    current_label = initial_label
    save_dir = os.path.join(DATA_DIR, "train", current_label)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    captured_count = 0
    flash_until = 0  # 캡처 시 화면 깜빡임 효과 타이머

    print("\n" + "=" * 50)
    print("  웹캠 데이터 수집 도구")
    print("=" * 50)
    print(f"  현재 클래스: {CLASS_KO[current_label]} ({current_label})")
    print(f"  저장 경로  : {save_dir}")
    print()
    print("  [Space/Enter] 캡처  |  [1~5] 클래스 전환  |  [Q] 종료")
    print("=" * 50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w = display.shape[:2]
        color = CLASS_COLOR.get(current_label, (255, 255, 255))

        # ── 상단 정보 바 ──
        cv2.rectangle(display, (0, 0), (w, 70), (30, 30, 30), -1)

        # 클래스 뱃지
        label_text = f"[{current_label.upper()}] {CLASS_KO[current_label]}"
        cv2.putText(display, label_text, (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # 캡처 수
        count_text = f"Captured: {captured_count}"
        cv2.putText(display, count_text, (w - 250, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        # ── 하단 안내 바 ──
        cv2.rectangle(display, (0, h - 40), (w, h), (30, 30, 30), -1)
        help_text = "[Space] Capture  |  [1]Plastic [2]Can [3]Paper [4]Glass [5]Trash  |  [Q] Quit"
        cv2.putText(display, help_text, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # ── 테두리 색상 (현재 클래스 표시) ──
        cv2.rectangle(display, (0, 70), (w - 1, h - 40), color, 3)

        # ── 캡처 플래시 효과 ──
        if time.time() < flash_until:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), -1)
            alpha = 0.3
            display = cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)
            cv2.putText(display, "SAVED!", (w // 2 - 80, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

        cv2.imshow("Data Capture Tool", display)

        key = cv2.waitKey(1) & 0xFF

        # ── 캡처: Space(32) 또는 Enter(13) ──
        if key in (32, 13):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"webcam_{current_label}_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            captured_count += 1
            flash_until = time.time() + 0.3
            print(f"  [{captured_count}] 저장: {filepath}")

        # ── 클래스 전환: 1~5 ──
        elif chr(key) in CLASSES if key < 128 else False:
            new_label = CLASSES[chr(key)]
            if new_label != current_label:
                current_label = new_label
                save_dir = os.path.join(DATA_DIR, "train", current_label)
                os.makedirs(save_dir, exist_ok=True)
                print(f"\n  >> 클래스 전환: {CLASS_KO[current_label]} → 저장 경로: {save_dir}")

        # ── 종료: Q 또는 ESC ──
        elif key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"  수집 종료! 총 {captured_count}장 저장되었습니다.")
    print(f"  다음 단계: python src/train.py --epochs 5 --no_freeze --lr 0.00005")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="웹캠 데이터 수집 도구")
    parser.add_argument("--label", type=str, default="plastic",
                        choices=["plastic", "can", "paper", "glass", "trash"],
                        help="저장할 클래스 (기본: plastic)")
    args = parser.parse_args()
    main(args.label)
