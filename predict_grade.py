import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
MODEL_PATH = './results/task3_final_model_full_data.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

TARGET_FOLDERS_AND_COUNT = {
    r'C:\Users\jyr30\Desktop\fruit_freshness_project\data\test\rottenoranges': 4,
    r'C:\Users\jyr30\Desktop\fruit_freshness_project\data\test\freshapples': 4,
    r'C:\Users\jyr30\Desktop\fruit_freshness_project\data\test\freshbanana': 4,
}

# ----------------------------------------------------------------------
# 모델 정의
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

# ----------------------------------------------------------------------
# 전처리
# ----------------------------------------------------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

FRUIT_CLASSES = ['사과', '바나나', '오렌지']
FRESHNESS_CLASSES = ['신선함 (Fresh)', '썩음 (Rotten)']
GRADE_CLASSES = {
    0: '0등급 (매우 신선)', 1: '1등급 (신선)', 2: '2등급 (약간 변색)', 
    3: '3등급 (상당히 썩음)', 4: '4등급 (매우 심하게 썩음)'
}

# 한글 폰트 설정
import matplotlib.font_manager as fm
font_path = 'C:/Windows/Fonts/malgunbd.ttf'
if os.path.exists(font_path):
    font_name = fm.FontProperties(fname=font_path, size=10).get_name()
    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------------------------
# 이미지 랜덤 선택
# ----------------------------------------------------------------------
def get_random_image_paths(folders_and_counts):
    all_paths = []
    for folder_path, count in folders_and_counts.items():
        if not os.path.isdir(folder_path):
            print(f"폴더 없음: {folder_path}")
            continue

        valid_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        num_to_select = min(count, len(valid_files))
        selected_files = random.sample(valid_files, num_to_select)

        for file_name in selected_files:
            all_paths.append(os.path.join(folder_path, file_name))

    return all_paths

# ----------------------------------------------------------------------
# 예측 + 팝업(안 멈춤 버전)
# ----------------------------------------------------------------------
def predict_and_show_random():

    image_paths = get_random_image_paths(TARGET_FOLDERS_AND_COUNT)
    if not image_paths:
        print(" 예측할 이미지 없음")
        return

    print("모델 로드 중...")
    model = MultiTaskFruitNet().to(device)

    if not os.path.exists(MODEL_PATH):
        print(f" 모델 없음: {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(" 모델 로드 완료 — 예측 시작!")

    plt.ion()  # 인터랙티브 모드 활성화 (응답없음 방지)

    for i, image_path in enumerate(image_paths):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = test_transform(image).unsqueeze(0).to(device)
        except:
            print(f" 이미지 불러오기 실패: {image_path}")
            continue

        with torch.no_grad():
            fruit_out, freshness_out, grade_out = model(image_tensor)

        fruit_pred = fruit_out.argmax(1).item()
        freshness_pred = freshness_out.argmax(1).item()
        grade_pred = grade_out.argmax(1).item()

        prediction_text = (
            f"과일: {FRUIT_CLASSES[fruit_pred]}\n"
            f"신선도: {FRESHNESS_CLASSES[freshness_pred]}\n"
            f"등급: {GRADE_CLASSES[grade_pred]}"
        )

        # 팝업 표시 (응답없음 방지)
        plt.figure(figsize=(6, 7))
        ax = plt.gca()
        ax.imshow(image)
        ax.set_title(f"[{i+1}/{len(image_paths)}] {os.path.basename(image_path)}")
        ax.axis('off')

        plt.figtext(
            0.5, 0.05, prediction_text, wrap=True, horizontalalignment='center', fontsize=11,
            bbox={"facecolor": "lightcoral" if "rotten" in image_path.lower() else "lightgreen",
                  "alpha": 0.7, "pad": 5}
        )

        plt.show(block=False)   #  실행 멈추지 않음
        plt.pause(3)   # 3초 기다리기    
        plt.close()          

    print("\n 모든 이미지 예측 완료!")
    
    print("파일 실행됨!")   # 테스트 추가
if __name__ == "__main__":
    predict_and_show_random()

