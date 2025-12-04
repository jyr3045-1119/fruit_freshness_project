import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# --- [1] 설정값  ---
# ----------------------------------------------------------------------
MODEL_PATH = './results/task3_final_model_full_data.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


IMAGE_PATHS_TO_PREDICT = [
    r'C:\Users\jyr30\Desktop\fruit_freshness_project\data\test\rottenapples\rotated_by_15_Screen Shot 2018-06-08 at 5.07.32 PM.png',
    r'C:\Users\jyr30\Desktop\fruit_freshness_project\data\test\freshapples\rotated_by_15_Screen Shot 2018-06-08 at 5.10.29 PM.png',
    r'C:\Users\jyr30\Desktop\fruit_freshness_project\data\test\rottenoranges\rotated_by_60_Screen Shot 2018-06-12 at 11.33.16 PM.png',
    # 필요한 만큼 더 추가
]


FRUIT_CLASSES = ['사과', '바나나', '오렌지']
FRESHNESS_CLASSES = ['신선함 (Fresh)', '썩음 (Rotten)']
# ----------------------------------------------------------------------




class MultiTaskFruitNet(nn.Module):
    
    def __init__(self, num_fruit_classes=3, num_freshness_classes=2, num_grade_classes=5):
        super(MultiTaskFruitNet, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Identity() 
        self.backbone = resnet
        
        self.fruit_head = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_fruit_classes)
        )
        self.freshness_head = nn.Sequential(
            nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_freshness_classes)
        )
        self.grade_head = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_grade_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        fruit_output = self.fruit_head(features)
        freshness_output = self.freshness_head(features)
        grade_output = self.grade_head(features)
        return fruit_output, freshness_output, grade_output

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------------------------------------------------------------
# --- [3] 예측 및 시각화 함수 (Task 3 제외) ---
# ----------------------------------------------------------------------

def predict_baseline(image_paths):
    print(f" 모델 로드 중...")
    if not os.path.exists(MODEL_PATH):
        print(f" 오류: 모델 파일 '{MODEL_PATH}'을(를) 찾을 수 없습니다.")
        return

    # 1. 모델 초기화 및 가중치 로드
    model = MultiTaskFruitNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() 
    

    import matplotlib.font_manager as fm
    font_path = 'C:/Windows/Fonts/malgunbd.ttf'
    if os.path.exists(font_path):
        font_name = fm.FontProperties(fname=font_path, size=10).get_name()
        plt.rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False 
    
    # 2. 각 이미지에 대해 예측 실행 및 개별 창 띄우기
    for i, image_path in enumerate(image_paths):
        # 1. 예측
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = test_transform(image).unsqueeze(0).to(device)
        except FileNotFoundError:
            print(f" 오류: 이미지 파일 '{image_path}'을(를) 찾을 수 없습니다. 건너뜁니다.")
            continue
        
        with torch.no_grad():
        
            fruit_output, freshness_output, _ = model(image_tensor) 

        fruit_pred = fruit_output.argmax(1).item()
        freshness_pred = freshness_output.argmax(1).item()
        
    
        prediction_text = (
            f"과일 이름: {FRUIT_CLASSES[fruit_pred]}\n"
            f"신선도: {FRESHNESS_CLASSES[freshness_pred]}\n"
          
        )
        
        plt.figure(figsize=(6, 7))
        ax = plt.gca()
        
        ax.imshow(image)
        ax.set_title(f"[{i+1}/{len(image_paths)}] 초기 모델 예측 결과", fontsize=14)
        ax.axis('off')
        
        plt.figtext(0.5, 0.05, prediction_text, wrap=True, horizontalalignment='center', fontsize=12, 
                    bbox={"facecolor":"#ADD8E6", "alpha":0.9, "pad":5}) # 하늘색 배경으로 한계점 강조

        print(f"\n[{i+1}/{len(image_paths)}] 예측 완료: {os.path.basename(image_path)}")
        plt.show(block=True) 
        
    print("\n✅ 초기 모델 예측 시뮬레이션 완료.")


if __name__ == '__main__':
    print("=" * 70)
    print(" 초기 모델 예측 결과 시뮬레이션 (Task 1 & 2만)")
    print("=" * 70)
    predict_baseline(IMAGE_PATHS_TO_PREDICT)
    print("=" * 70)