import os
import random
from typing import Tuple, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class ContaminationAugmentation:
    """오염된 상태를 시뮬레이션하기 위한 데이터 증강 클래스"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob
        # 강한 색상/밝기 왜곡 (음식물 얼룩, 찌든 때 모사)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.5,
            hue=0.1
        )
        # 지우기 기법으로 형태 훼손 모사 (찌그러짐, 가려짐 등)
        self.random_erasing = transforms.RandomErasing(
            p=1.0, 
            scale=(0.02, 0.15), 
            ratio=(0.3, 3.3), 
            value='random'
        )

    def __call__(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        확률적으로 증강을 적용하고, 오염 여부(is_contaminated) 라벨을 리턴합니다.
        
        Args:
            img_tensor: [C, H, W] 형태의 깨끗한 이미지 텐서
            
        Returns:
            (augmented_tensor, is_contaminated)
            is_contaminated: 1 (증강 적용/오염됨), 0 (증강 미적용/깨끗함)
        """
        # img_tensor는 이미 normalization 전의 float tensor (0.0 ~ 1.0) 이라고 가정
        
        if random.random() < self.prob:
            # 1. 색상 왜곡 (전체적인 더러움)
            # ColorJitter는 PIL 이미지나 uint8 텐서를 기대하는 경우가 많아,
            # float tensor 구조에서는 바로 적용 가능 (PyTorch 버전에 따라 다름)
            aug_tensor = self.color_jitter(img_tensor)
            
            # 2. 패치 얼룩 추가 (RandomErasing)
            # 개수(1~3개) 랜덤
            num_patches = random.randint(1, 3)
            for _ in range(num_patches):
                aug_tensor = self.random_erasing(aug_tensor)
                
            return aug_tensor, 1 # 오염됨
        else:
            return img_tensor, 0 # 깨끗함


class RecyclingDualHeadDataset(Dataset):
    """
    듀얼 헤드 모델을 위한 재활용 분류 데이터셋.
    기존 단일 폴더 구조(재질별 폴더)를 읽어와서 학습 중 동적으로 '오염' 라벨을 생성합니다.
    """
    def __init__(self, root_dir: str, class_to_idx: Dict[str, int], img_size: int = 224, is_train: bool = True):
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.is_train = is_train
        
        # 이미지 파일 목록 수집
        self.image_paths = []
        self.labels = [] # 재질 라벨
        
        for cls_name, cls_idx in class_to_idx.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                print(f"경고: 클래스 폴더가 존재하지 않습니다 ({cls_dir})")
                continue
                
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(cls_idx)
                    
        # 기본 전처리 (크기 조절 및 텐서 변환)
        self.base_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), # 0.0 ~ 1.0
        ])
        
        # 훈련 시에만 형태 변환(Flip/Rotation) 및 오염 합성 로직 적용
        if self.is_train:
            self.geometric_tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ])
            self.contamination_aug = ContaminationAugmentation(prob=0.5)
        else:
            self.contamination_aug = ContaminationAugmentation(prob=0.0) # 검증은 깨끗한 상태로만 테스트 혹은 별도 처리
            
        # 정규화 (모델 입력용 최종 단계)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        img_path = self.image_paths[idx]
        mat_label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {img_path} ({e})")
            # 에러 발생 시 더미 텐서 반환
            tensor = torch.zeros(3, 224, 224)
            return self.normalize(tensor), (mat_label, 0)
            
        tensor = self.base_tf(image)
        
        if self.is_train:
            tensor = self.geometric_tf(tensor)
            tensor, is_contam = self.contamination_aug(tensor)
        else:
            is_contam = 0 # Val 셋은 기본적으로 제공된 깨끗한 이미지라고 가정
            # 평가 단계에서 오염셋을 평가하고 싶다면 설계 변경 필요
            
        tensor = self.normalize(tensor)
        
        # (이미지, (재질번호, 오염여부)) 반환
        return tensor, (mat_label, is_contam)
