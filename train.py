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

# ----------------------------------------------------------------------
# --- [1] 설정값 
# ----------------------------------------------------------------------

# 훈련 데이터의 루트 경로
BASE_TRAIN_DATA_DIR = r'C:\Users\jyr30\Desktop\fruit_freshness_project\data\train'

# 라벨 CSV 파일명 
CSV_PATH = 'custom_task3_labels.csv' 


NUM_EPOCHS = 20
BATCH_SIZE = 32
# Task 3 (등급)에 더 많은 가중치를 부여하여 순서 학습에 집중
TASK_WEIGHTS = {'fruit': 1.0, 'freshness': 1.0, 'grade': 2.0} 


# ----------------------------------------------------------------------
# --- [2] Custom Loss Function 정의 (학습 알고리즘 변형의 핵심) ---
# ----------------------------------------------------------------------

def custom_cost_sensitive_loss(predictions, targets, num_classes=5):
    """
    Task 3: 썩음 정도 등급 분류를 위한 Cost-Sensitive Loss.
    예측 등급과 실제 등급의 차이(|예측-실제|)가 클수록 Loss에 페널티를 부여합니다.
    """
    # 1. 일반 CrossEntropy Loss 계산 (reduction='none'으로 배치별 손실 유지)
    ce_loss = nn.CrossEntropyLoss(reduction='none')(predictions, targets)
    
    # 2. 예측된 클래스 인덱스 (가장 높은 확률)를 가져옵니다.
    _, predicted_classes = torch.max(predictions, 1)
    
    # 3. 등급 차이 절대값 계산: |예측 등급 - 실제 등급|
    cost_factor = torch.abs(predicted_classes.float() - targets.float())
    
    # 4. Cost-Sensitive Loss 계산: CE Loss에 Cost Factor를 곱하여 페널티 부과
    # (1.0 + cost_factor)를 곱하여 Cost가 0일 때도 CE Loss가 유지되도록 함
    custom_loss = ce_loss * (1.0 + cost_factor) 
    
    return custom_loss.mean()


# ----------------------------------------------------------------------
# --- [3] Custom Dataset 클래스 정의 ---
# ----------------------------------------------------------------------

class FruitMultiTaskDataset(Dataset):
    def __init__(self, data_dir, csv_path, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = self._get_transforms(mode)
        
        # 1. Task 3 (등급) 레이블 로드 및 파일 이름으로 딕셔너리 생성
        self.task3_labels = self._load_task3_labels(csv_path)
        
        # 2. Task 3 라벨을 가진 데이터 경로만 필터링하여 최종 데이터 리스트 생성 (200개)
        self.final_data = self._load_data_paths_with_task3()
        
        # 데이터가 없을 경우 최종 오류 발생
        if not self.final_data:
            raise FileNotFoundError("Task 3 라벨이 있는 파일을 원본 데이터 폴더에서 찾을 수 없습니다. (200개 매칭 실패)")
            
        print(f"Dataset initialized ({mode}): {len(self.final_data)} samples for multi-task learning.")

    def _get_transforms(self, mode):
        # 학습을 위한 데이터 증강(Augmentation) 및 정규화
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
        # CSV 파일 로드 (이 시점에서 FileNotFoundError는 train_model에서 처리됨)
        df = pd.read_csv(csv_path)
        label_dict = {}
        
        for _, row in df.iterrows():
            # CSV의 경로에서 파일 이름만 추출 
            filename_only = os.path.basename(row['image_path']) 
            grade = int(row['grade_label']) 
            
            # 파일 이름:등급 딕셔너리 생성 (매칭 키)
            label_dict[filename_only] = grade
            
        return label_dict

    def _load_data_paths_with_task3(self):
        data = []
        base_train_data = self.data_dir # train.py에서 전달받은 BASE_TRAIN_DATA_DIR 사용
        
        for fruit_type in ['apples', 'banana', 'oranges']:
            for freshness in ['fresh', 'rotten']:
          
                dir_path = os.path.join(base_train_data, f'{freshness}{fruit_type}')
                
                if not os.path.exists(dir_path):
                    continue
                
                for filename in os.listdir(dir_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # 원본 데이터 파일 이름
                        filename_only = os.path.basename(filename) 

                        # 파일 이름이 Task 3 라벨 Dict에 있는지 확인 (200개 샘플만 추출)
                        if filename_only in self.task3_labels: 
                            
                            # Task 1: 과일 종류 (0: apple, 1: banana, 2: orange)
                            fruit_label = 0 if 'apples' in fruit_type else (1 if 'banana' in fruit_type else 2)
                            
                            # Task 2: 신선도 (0: fresh, 1: rotten)
                            freshness_label = 0 if freshness == 'fresh' else 1
                            
                            # Task 3: 등급 (CSV에서 가져옴)
                            grade_label = self.task3_labels[filename_only] 
                            
                            data.append({
                                'full_path': os.path.join(dir_path, filename),
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
# --- [4] Custom Multi-Task ) ---
# ----------------------------------------------------------------------

class MultiTaskFruitNet(nn.Module):
    def __init__(self, num_fruit_classes=3, num_freshness_classes=2, num_grade_classes=5):
        super(MultiTaskFruitNet, self).__init__()
        
        # 1. 백본 로드 (Feature Extractor) - ResNet-18 사용
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 마지막 FC 층 제거 (분류 Head 제거)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Identity() 
        
        self.backbone = resnet
        
        # 2. 병렬 Head 1: Task 1 - 과일 종류 분류 (3 classes)
        self.fruit_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_fruit_classes)
        )
        
        # 3. 병렬 Head 2: Task 2 - 신선도 분류 (2 classes)
        self.freshness_head = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_freshness_classes)
        )
        
        # 4. 병렬 Head 3: Task 3 - 썩음 정도 등급 분류 (5 classes)
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
# --- [5] Main 학습 루프 ---
# ----------------------------------------------------------------------

def train_model():
    # GPU / CPU 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # 데이터셋 및 DataLoader 준비
        train_dataset = FruitMultiTaskDataset(BASE_TRAIN_DATA_DIR, CSV_PATH, mode='train')
    except FileNotFoundError as e:
        # CSV 파일 또는 데이터 파일 경로 오류 시 메시지 출력
        print(f"  데이터 로드 중 치명적 오류: {e}")
        print(f"   - CSV 파일 경로 확인: {CSV_PATH}")
        print(f"   - 원본 데이터 경로 확인: {BASE_TRAIN_DATA_DIR}")
        return

    # 데이터 로더 설정
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # 모델 초기화
    model = MultiTaskFruitNet().to(device)
    
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
            targets_grade = targets['grade'].to(device)

            optimizer.zero_grad()
            
            output_fruit, output_freshness, output_grade = model(images)
            
            # --- Loss 계산 (Multi-Task Loss) ---
            
            loss_fruit = criterion_ce(output_fruit, targets_fruit) * TASK_WEIGHTS['fruit']
            loss_freshness = criterion_ce(output_freshness, targets_freshness) * TASK_WEIGHTS['freshness']
            
            # Custom Cost-Sensitive Loss 적용 (Task 3 핵심)
            loss_grade = custom_cost_sensitive_loss(output_grade, targets_grade) * TASK_WEIGHTS['grade']
            
            # 전체 Loss 합산
            loss = loss_fruit + loss_freshness + loss_grade
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Avg Total Loss: {avg_loss:.4f}")

    print("\n--- 학습 완료 ---")
    
    MODEL_SAVE_PATH = './results/task3_warmup_model.pth' # 다음 학습을 위한 초기 가중치로 사용
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True) 
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f" 모델 가중치 저장 완료: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
    
   