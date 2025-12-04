import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np

# 훈련 데이터의 루트 경로
BASE_TRAIN_DATA_DIR = r'C:\Users\jyr30\Desktop\fruit_freshness_project\data\train'

CSV_PATH = 'custom_task3_labels.csv' 

# ----------------------------------------------------------------------
# --- [1-2] 최종 학습 하이퍼파라미터 ---

NUM_EPOCHS = 30 
BATCH_SIZE = 32
TASK_WEIGHTS = {'fruit': 1.0, 'freshness': 1.0, 'grade': 2.0} 
WARMUP_WEIGHTS_PATH = './results/task3_warmup_model.pth' # 이전 단계에서 저장된 가중치

# ----------------------------------------------------------------------
# ---  Custom Loss Function 정의  ---
# ----------------------------------------------------------------------

def custom_cost_sensitive_loss(predictions, targets, num_classes=5):
    """
    Task 3: 썩음 정도 등급 분류를 위한 Cost-Sensitive Loss.
    """
    # targets이 -1인 경우, 마스크를 생성하지 않고 CrossEntropyLoss의 ignore_index에 의존합니다.
    mask = (targets != -1)
    
    # 1. 일반 CrossEntropy Loss 계산 (ignore_index=-1로 설정!)
    # 이 인자를 추가하면 -1을 무시하고 Loss를 계산합니다.
    ce_loss_full = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)(predictions, targets)
    
    # 마스크가 True인 샘플들만 가져와 Cost-Sensitive Loss 계산에 사용
    ce_loss = ce_loss_full[mask]
    
    # 마스크된 텐서에 데이터가 없으면 Loss 0 반환
    if ce_loss.numel() == 0:
        return torch.tensor(0.0, device=predictions.device)
        
    # 2. 예측된 클래스 인덱스 (마스크된 예측만 사용)
    _, predicted_classes_full = torch.max(predictions, 1)
    predicted_classes = predicted_classes_full[mask]
    
    # 3. 등급 차이 절대값 계산
    masked_targets = targets[mask]
    cost_factor = torch.abs(predicted_classes.float() - masked_targets.float())
    
    # 4. Cost-Sensitive Loss 계산 및 평균
    custom_loss = ce_loss * (1.0 + cost_factor)
    
    return custom_loss.mean()

# ----------------------------------------------------------------------
# --- [3] Custom Dataset 클래스 정의  ---
# ----------------------------------------------------------------------

class FruitMultiTaskDataset(Dataset):
    def __init__(self, data_dir, csv_path, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = self._get_transforms(mode)
        
        # 1. Task 3 (등급) 레이블 로드 및 파일 이름으로 딕셔너리 생성
        self.task3_labels = self._load_task3_labels(csv_path)
        
        # 2. 모든 데이터 경로를 로드하여 최종 데이터 리스트 생성
        self.final_data = self._load_all_data_paths()
        
        if not self.final_data:
            raise FileNotFoundError("원본 데이터 폴더에서 파일을 찾을 수 없습니다.")
            
        print(f"Dataset initialized ({mode}): {len(self.final_data)} samples loaded for final multi-task learning.")

    def _get_transforms(self, mode):
       
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _load_task3_labels(self, csv_path):
       
        try:
            df = pd.read_csv(csv_path)
            label_dict = {}
            for _, row in df.iterrows():
                filename_only = os.path.basename(row['image_path'])  
                grade = int(row['grade_label']) 
                label_dict[filename_only] = grade
            return label_dict
        except FileNotFoundError:
             print(f"Warning: CSV file '{csv_path}' not found. Task 3 labels will be treated as missing.")
             return {}

    def _load_all_data_paths(self):
        data = []
        base_train_data = self.data_dir
        
        # 모든 하위 폴더를 반복하며 전체 데이터 로드
        for fruit_type in ['apples', 'banana', 'oranges']:
            for freshness in ['fresh', 'rotten']:
                dir_path = os.path.join(base_train_data, f'{freshness}{fruit_type}')
                
                if not os.path.exists(dir_path):
                    continue
                
                for filename in os.listdir(dir_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(dir_path, filename)
                        filename_only = os.path.basename(filename) 

                        # Task 1: 과일 종류 (0: apple, 1: banana, 2: orange)
                        fruit_label = 0 if 'apples' in fruit_type else (1 if 'banana' in fruit_type else 2)
                        
                        # Task 2: 신선도 (0: fresh, 1: rotten)
                        freshness_label = 0 if freshness == 'fresh' else 1
                        
                        # Task 3: 등급 - CSV에 있으면 해당 라벨 사용, 없으면 -1 (Loss 계산에서 제외할 값) 할당
                        grade_label = self.task3_labels.get(filename_only, -1) 
                        
                        data.append({
                            'full_path': full_path,
                            'fruit_label': fruit_label,
                            'freshness_label': freshness_label,
                            'grade_label': grade_label
                        })
        return data

    def __len__(self):
        return len(self.final_data)

    def __getitem__(self, idx):
        item = self.final_data[idx]
        
        image = Image.open(item['full_path']).convert('RGB')
        image = self.transform(image)
        
        targets = {
            'fruit': torch.tensor(item['fruit_label'], dtype=torch.long),
            'freshness': torch.tensor(item['freshness_label'], dtype=torch.long),
            'grade': torch.tensor(item['grade_label'], dtype=torch.long)
        }
        
        return image, targets


# ----------------------------------------------------------------------
# --- [4] Custom Multi-Task
# ----------------------------------------------------------------------

class MultiTaskFruitNet(nn.Module):
    def __init__(self, num_fruit_classes=3, num_freshness_classes=2, num_grade_classes=5):
        super(MultiTaskFruitNet, self).__init__()
        
        # 1. 백본 로드 (ResNet-18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Identity() 
        self.backbone = resnet
        
        # 2. 병렬 Head 1: Task 1 - 과일 종류 분류
        self.fruit_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_fruit_classes)
        )
        
        # 3. 병렬 Head 2: Task 2 - 신선도 분류
        self.freshness_head = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_freshness_classes)
        )
        
        # 4. 병렬 Head 3: Task 3 - 썩음 정도 등급 분류
        self.grade_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_grade_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        fruit_output = self.fruit_head(features)
        freshness_output = self.freshness_head(features)
        grade_output = self.grade_head(features)
        
        return fruit_output, freshness_output, grade_output


# ----------------------------------------------------------------------
# --- [5] Main 학습 루프 (가중치 로드 및 Loss 처리 수정) ---
# ----------------------------------------------------------------------

def train_model():
    # GPU / CPU 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # 데이터셋 및 DataLoader 준비 
        train_dataset = FruitMultiTaskDataset(BASE_TRAIN_DATA_DIR, CSV_PATH, mode='train')
    except FileNotFoundError as e:
        print(f" 데이터 로드 중 치명적 오류: {e}")
        return

    # 데이터 로더 설정 (이제 데이터 수가 많아지므로 num_workers는 환경에 따라 설정)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # 모델 초기화
    model = MultiTaskFruitNet().to(device)

    # -------------------------------------------------------------
    #  Warm-up 가중치 로드 
    if os.path.exists(WARMUP_WEIGHTS_PATH):
        print(f"⭐ WARM-UP 가중치 로드: {WARMUP_WEIGHTS_PATH}")
        model.load_state_dict(torch.load(WARMUP_WEIGHTS_PATH, map_location=device))
    else:
        print(" WARM-UP 가중치 파일을 찾을 수 없습니다. (랜덤 초기화로 시작)")
    # -------------------------------------------------------------
    
    # 옵티마이저 정의
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Task 1, 2는 일반 CrossEntropy Loss 사용
    criterion_ce = nn.CrossEntropyLoss()
    
    # 학습 루프 시작
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Training")
        
        for images, targets in train_loop:
            images = images.to(device)
            
         
            targets_fruit = targets['fruit'].to(device)
            targets_freshness = targets['freshness'].to(device)
            targets_grade = targets['grade'].to(device) # -1 포함

            optimizer.zero_grad()
            
            output_fruit, output_freshness, output_grade = model(images)
            
            # --- Loss 계산 (Multi-Task Loss) ---
            
            # Task 1 & 2는 전체 데이터에 대해 계산
            loss_fruit = criterion_ce(output_fruit, targets_fruit) * TASK_WEIGHTS['fruit']
            loss_freshness = criterion_ce(output_freshness, targets_freshness) * TASK_WEIGHTS['freshness']
            
            #  Custom Cost-Sensitive Loss 적용 (Task 3 핵심)
            # Loss 함수 내부에서 targets_grade의 -1 값을 제외하고 계산됩니다.
            loss_grade = custom_cost_sensitive_loss(output_grade, targets_grade) * TASK_WEIGHTS['grade']
            
            # 전체 Loss 합산
            loss = loss_fruit + loss_freshness + loss_grade
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Avg Total Loss: {avg_loss:.4f}")

    print("\n--- 최종 학습 완료 ---")
    
    # 6. 최종 모델 저장
    FINAL_MODEL_SAVE_PATH = './results/task3_final_model_full_data.pth'
    os.makedirs(os.path.dirname(FINAL_MODEL_SAVE_PATH), exist_ok=True) 
    torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
    print(f" 최종 모델 가중치 저장 완료: {FINAL_MODEL_SAVE_PATH}")


if __name__ == '__main__':
    train_model()