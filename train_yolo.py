"""
YOLO 단독 학습 스크립트.

사용법:
    python train_yolo.py                                    # 기본 설정으로 학습
    python train_yolo.py --model yolo11s.pt --epochs 50     # 모델/에포크 변경
    python train_yolo.py --seed 123 --name test_run         # 시드/실험명 변경
"""

import argparse
import random

import numpy as np
import torch
from ultralytics import YOLO


# =========================================================
# 1. 랜덤 시드 설정
# =========================================================

def set_random_seed(seed_value: int = 42):
    """재현성을 위한 랜덤 시드 고정."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 2. CLI 인자 파싱
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 학습 스크립트")

    # 모델 & 데이터
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="YOLO 모델 가중치 (default: yolo11m.pt)")
    parser.add_argument("--data", type=str, default="data/99_exp_total_dataset/data.yaml", help="데이터 YAML 경로")

    # 학습 기본
    parser.add_argument("--epochs", type=int, default=150, help="학습 에포크 수 (default: 150)")
    parser.add_argument("--batch", type=int, default=16, help="배치 크기 (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="입력 이미지 크기 (default: 640)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (default: 42)")

    # 저장 경로
    parser.add_argument("--project", type=str, default="./runs", help="결과 저장 기본 디렉토리 (default: ./runs)")
    parser.add_argument("--name", type=str, default="yolov11m", help="실험 이름 (project 하위 폴더)")

    # 옵티마이저 & 스케줄러
    parser.add_argument("--optimizer", type=str, default="AdamW", help="옵티마이저 (default: AdamW)")
    parser.add_argument("--lr0", type=float, default=0.001, help="초기 학습률 (default: 0.001)")
    parser.add_argument("--lrf", type=float, default=0.01, help="최종 LR 비율 (default: 0.01)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--warmup_epochs", type=int, default=None, help="Warmup 에포크 (default: 전체의 5%%)")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience (default: 40)")
    parser.add_argument("--workers", type=int, default=12, help="Dataloader 워커 수 (default: 12)")
    parser.add_argument("--cos_lr", type=bool, default=True, help="코사인 LR 스케줄러 사용 (default: True)")

    return parser.parse_args()


# =========================================================
# 3. 메인 학습 함수
# =========================================================

def main():
    args = parse_args()

    # 시드 설정
    set_random_seed(args.seed)

    # Warmup 에포크 기본값: 전체 에포크의 5%
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else int(0.05 * args.epochs)

    # 모델 로드
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    # 학습 파라미터 구성
    train_params = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "cos_lr": args.cos_lr,
        "warmup_epochs": warmup_epochs,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "workers": args.workers,
        "patience": args.patience,
        # 데이터 증강 (HF와 공정한 비교를 위해 고정)
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.9,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
        "erasing": 0.2,
    }

    print("=" * 60)
    print("  YOLO Training Configuration")
    print("=" * 60)
    for k, v in train_params.items():
        print(f"  {k:20s}: {v}")
    print("=" * 60)

    # 학습 실행
    results = model.train(**train_params)

    print("\n" + "=" * 60)
    print(f"  학습 완료! 결과: {args.project}/{args.name}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
