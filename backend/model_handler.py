import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import time
from ultralytics import YOLO
import timm
from torchvision import transforms

# --- DualHead Architecture (Same as train.py/inference.py) ---
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

class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 프로젝트 루트 경로 (backend 폴더의 상위 폴더)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Paths (Root 기준 절대 경로로 설정)
        self.model_path = os.path.join(self.base_dir, "models", "best_model.pth")
        self.config_path = os.path.join(self.base_dir, "models", "model_config.json")
        
        # Load Config
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        # Load Classification Model
        self.classifier = DualHeadMobileNetV3(num_material_classes=self.config["num_classes"])
        if os.path.exists(self.model_path):
            try:
                # strict=False를 사용하여 클래스 개수 변경 시에도 로딩 가능하게 함 (재학습 전까지는 부정확할 수 있음)
                self.classifier.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
            except Exception as e:
                print(f"Warning: Could not load model weights strictly: {e}")
        self.classifier.to(self.device)
        self.classifier.eval()
        
        # Load YOLOv8-Nano (Detection)
        # Note: 'yolov8n.pt' will be downloaded automatically if not present
        self.detector = YOLO("yolov8n.pt")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def analyze(self, image: Image.Image):
        start_time = time.time()
        
        # 1. Detection Stage
        # Predict on the full image
        results = self.detector.predict(image, conf=0.5, verbose=False)
        
        predictions = []
        detected = False
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # 이미지 중앙 좌표 계산
            img_w, img_h = image.size
            center_x, center_y = img_w / 2, img_h / 2
            
            # 모든 탐지 객체 중 중앙에서 가장 가까운 하나만 선택
            best_box = None
            min_dist = float('inf')
            
            for box in results[0].boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                box_center_x = (bx1 + bx2) / 2
                box_center_y = (by1 + by2) / 2
                
                dist = ((box_center_x - center_x)**2 + (box_center_y - center_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_box = (bx1, by1, bx2, by2)
            
            if best_box:
                detected = True
                x1, y1, x2, y2 = best_box
                
                # 2. Classification Stage (Crop & Classify) - 가장 중앙의 객체 하나만 처리
                cropped_img = image.crop((x1, y1, x2, y2))
                tensor = self.transform(cropped_img.convert("RGB")).unsqueeze(0).to(self.device)
                
                mat_logits, contam_logits = self.classifier(tensor)
                
                # Probs
                mat_probs = F.softmax(mat_logits, dim=1)[0]
                contam_prob = torch.sigmoid(contam_logits)[0].item()
                
                # Material Result
                top_idx = mat_probs.argmax().item()
                label_en = self.config["class_names"][top_idx]
                label_ko = self.config["class_ko"].get(label_en, label_en)
                mat_conf = mat_probs[top_idx].item()
                
                # Contamination Result
                is_contaminated = contam_prob >= 0.5
                contam_status = "contaminated" if is_contaminated else "clean"
                
                # Final Recyclable Logic (from inference.py)
                is_mat_recyclable = self.config["recyclable"].get(label_en, False)
                final_recyclable = is_mat_recyclable and (not is_contaminated)
                
                predictions.append({
                    "label": label_en,
                    "korean_label": label_ko,
                    "confidence": float(mat_conf),
                    "is_recyclable": final_recyclable,
                    "contamination_status": contam_status,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })

        inference_time_ms = (time.time() - start_time) * 1000
        
        return {
            "detected": detected,
            "predictions": predictions,
            "inference_time_ms": round(inference_time_ms, 2)
        }
