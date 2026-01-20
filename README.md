# 🚀 HF Object Detection Trainer
    🚧 Work In Progress (WIP)

    This project is currently under active development. The code is not yet complete and features will be updated gradually. Please note that the structure or APIs may change.

    (현재 코드가 완성되지 않았으며, 기능이 차근차근 업데이트될 예정입니다. 사용에 유의해 주세요.)

---
**Hugging Face Transformers 기반의 유연하고 확장 가능한 객체 탐지(Object Detection) 학습 파이프라인**

## 📖 Introduction (Motivation)
객체 탐지(Object Detection) 모델을 학습시킬 때, 우리는 종종 딜레마에 빠집니다.

- **YOLO (Ultralytics)**: 정말 훌륭하고 빠릅니다. 하지만 YOLO 구조에 특화되어 있어, 다른 최신 논문의 모델을 실험하거나 커스텀 아키텍처를 적용하기엔 제약이 많습니다.

- **MMDetection / Detectron2**: 기능이 강력하고 방대합니다. 하지만 그만큼 구조가 복잡하여 내가 원하는 기능만 골라 쓰기가 어렵습니다. 특히 라이브러리 버전 의존성(Dependency Hell)을 맞추는 환경 설정 단계부터 지치기 쉽습니다.

**"모델만 깔끔하게 불러오고, 나머지는 내가 원하는 대로 자유롭게 코딩할 수는 없을까?"**

이 프로젝트는 이러한 고민에서 시작되었습니다. Hugging Face Transformers API의 강력한 모델 호환성을 활용하여, 복잡한 프레임워크의 제약 없이 데이터셋, 모델 구조, Loss Function, Optimizer 등을 자유롭게 튜닝하고 실험할 수 있는 "가볍고 모듈화된 학습 템플릿"을 지향합니다.

## ✨ Features
- ⚡ **Hugging Face Ecosystem**: `AutoModel`을 활용하여 DETR, YOLOS, Table Transformer 등 Hugging Face Hub에 있는 다양한 객체 탐지 모델을 코드 수정 없이 바로 실험할 수 있습니다.

- 🛠️ **Fully Customizable**: 데이터셋 로더부터 Training Loop, Loss Function까지 모든 부분이 모듈화되어 있어 쉽게 커스터마이징이 가능합니다.

- 📂 **Custom Dataset Friendly**: 공개 데이터셋뿐만 아니라, 직접 구축한 커스텀 데이터셋(예: CVAT XML 포맷 등)을 쉽게 학습에 활용할 수 있는 가이드를 제공합니다.

- 🧱 **Modular Structure**: 설정(`configs`), 데이터(src/data), 모델(`src/model`), 실행(`train.py`)이 명확히 분리되어 있어 유지보수가 쉽습니다.

## 📂 Project Structure
```
hf-object-detection-trainer/
├── configs/
│   └── config.py          # 모델, 데이터 경로, 하이퍼파라미터 통합 관리
├── data/                  # 데이터셋 폴더 (ignored)
├── src/
│   ├── dataset.py         # 커스텀 데이터셋 파싱 및 로드 (CVAT XML 예제 포함)
│   ├── model.py           # Hugging Face 모델 및 프로세서 로더
│   └── utils.py           # Collate function, 시각화 도구 등
├── train.py               # 학습 실행 스크립트 (Training Loop)
├── evaluate.py            # 검증 및 테스트 스크립트 (Evaluation)
├── requirements.txt       # 필요 라이브러리
└── README.md
```

## 🚀 Getting Started
1. **Installation**
    ```
    # 레포지토리 클론
    git clone https://github.com/your-username/hf-object-detection-trainer.git
    cd hf-object-detection-trainer

    # 필수 라이브러리 설치
    pip install -r requirements.txt
    ```
2. **Configuration (`configs/config.py`)**

    `configs/config.py` 파일에서 데이터셋 경로와 사용할 모델을 지정합니다.
    ```
    class Config:
        # 데이터셋 경로 설정
        BASE_DIR = "./dataset_cvat"
        
        # 사용할 모델 (Hugging Face Hub ID)
        # 예: "facebook/detr-resnet-50", "hustvl/yolos-tiny" 등
        MODEL_CHECKPOINT = "facebook/detr-resnet-50"
        
        # 하이퍼파라미터 설정
        EPOCHS = 50
        BATCH_SIZE = 4
        LEARNING_RATE = 1e-5
        # ...
    ```
3. **Training**

    설정이 완료되면 아래 명령어로 학습을 시작합니다.
    ```
    python train.py
    ```
    학습이 완료되면 `./runs/` 폴더에 모델 가중치(`model.safetensors`)와 설정 파일이 저장됩니다.

4. **Evaluation**

    학습된 모델의 성능을 검증하거나 테스트하려면 다음을 실행합니다.
    ```
    python evaluate.py
    ```

## 🔧 Customization Guide
- **새로운 모델 사용**: `config.py`의 `MODEL_CHECKPOINT`만 변경하면 됩니다. (단, 데이터셋 포맷이 호환되는지 확인 필요)

- **데이터셋 포맷 변경**: `src/dataset.py`의 `__getitem__` 메서드에서 본인의 데이터 포맷(YOLO txt, COCO json 등)을 읽어 `processor`에 넘겨주도록 수정하세요.

- **Augmentation 추가**: `src/dataset.py` 내에 `albumentations` 등을 활용한 증강 로직을 추가할 수 있습니다.

## 📝 License
This project is licensed under the MIT License. See the LICENSE file for details.