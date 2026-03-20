"""
YOLO 반복 실험 자동화 스크립트.

사용법:
    python run_yolo_experiments.py                          # 전체 실험 (모든 모델 x 5회)
    python run_yolo_experiments.py --model yolo11m          # yolo11m만 5회
    python run_yolo_experiments.py --model yolo11s          # yolo11s만 5회
    python run_yolo_experiments.py --repeats 3              # 반복 횟수 변경
    python run_yolo_experiments.py --dry-run                # 실제 학습 없이 설정만 출력
"""

import argparse
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass


# =========================================================
# 1. 실험 설정 정의
# =========================================================

# 반복 실험마다 사용할 시드 목록 (HF 실험과 동일)
SEEDS = [42, 123, 456, 789, 1024]

# 모든 모델에 공통으로 적용할 학습 파라미터
COMMON_PARAMS = {
    "data": "data/99_exp_total_dataset/data.yaml",
    "epochs": 150,
    "batch": 16,
    "imgsz": 640,
    "optimizer": "AdamW",
    "lr0": 1e-4,
    "lrf": 0.01,
    "weight_decay": 1e-4,
    "patience": 40,
    "workers": 12,
    "cos_lr": True,

    # [데이터 증강 (HF와 공정한 비교를 위해 통제)]
    'hsv_h': 0.015,  # 참외 노란색 보존
    'hsv_s': 0.3,
    'hsv_v': 0.3,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.9,    # HF의 crop/affine 줌인/아웃과 유사한 강도
    'fliplr': 0.5,
    'mosaic': 0.0,   # 공정성을 위해 YOLO 필살기 끄기
    'mixup': 0.0,
    'erasing': 0.1
}


@dataclass
class ModelConfig:
    """모델별 실험 설정."""
    key: str                # 커맨드라인에서 사용할 짧은 이름
    weights: str            # YOLO 모델 가중치 파일
    project_name: str       # runs/{project_name}/ 하위에 결과 저장
    experiment_name: str    # 실험 폴더 접두사


# --- 모델별 설정 ---
MODEL_CONFIGS = {
    "rtdetr-l": ModelConfig (
        key="rtdetr-l",
        weights="rtdetr-l.pt",
        project_name="rtdetr-l",
        experiment_name="repeat-exp",
    ),
    "rtdetr-x": ModelConfig (
        key="rtdetr-x",
        weights="rtdetr-x.pt",
        project_name="rtdetr-x",
        experiment_name="repeat-exp",
    ),
    # "yolo11n": ModelConfig(
    #     key="yolo11n",
    #     weights="yolo11n.pt",
    #     project_name="yolo11n",
    #     experiment_name="repeat-exp",
    # ),
    # "yolo11s": ModelConfig(
    #     key="yolo11s",
    #     weights="yolo11s.pt",
    #     project_name="yolo11s",
    #     experiment_name="repeat-exp",
    # ),
    # "yolo11m": ModelConfig(
    #     key="yolo11m",
    #     weights="yolo11m.pt",
    #     project_name="yolo11m",
    #     experiment_name="repeat-exp",
    # ),
    # "yolo11l": ModelConfig(
    #     key="yolo11l",
    #     weights="yolo11l.pt",
    #     project_name="yolo11l",
    #     experiment_name="repeat-exp",
    # ),
    # "yolo11x": ModelConfig(
    #     key="yolo11x",
    #     weights="yolo11x.pt",
    #     project_name="yolo11x",
    #     experiment_name="repeat-exp",
    # ),
}


# =========================================================
# 2. 단일 실험 실행
# =========================================================

def build_command(model_cfg: ModelConfig, seed: int) -> list:
    """train_yolo.py 실행을 위한 커맨드라인 인자 목록 생성."""
    experiment_name = f"{model_cfg.experiment_name}_seed{seed}"

    cmd = [
        sys.executable, "train_yolo.py",
        "--model", model_cfg.weights,
        "--data", COMMON_PARAMS["data"],
        "--epochs", str(COMMON_PARAMS["epochs"]),
        "--batch", str(COMMON_PARAMS["batch"]),
        "--imgsz", str(COMMON_PARAMS["imgsz"]),
        "--seed", str(seed),
        "--project", f"./runs/{model_cfg.project_name}",
        "--name", experiment_name,
        "--optimizer", COMMON_PARAMS["optimizer"],
        "--lr0", str(COMMON_PARAMS["lr0"]),
        "--lrf", str(COMMON_PARAMS["lrf"]),
        "--weight_decay", str(COMMON_PARAMS["weight_decay"]),
        "--patience", str(COMMON_PARAMS["patience"]),
        "--workers", str(COMMON_PARAMS["workers"]),
    ]
    return cmd


# =========================================================
# 3. 메인 실행 로직
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="YOLO 반복 실험 자동화 스크립트")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default="all",
        help="실험할 모델 선택 (default: all)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="모델당 반복 횟수 (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 학습 없이 설정만 출력",
    )
    args = parser.parse_args()

    # 실행할 모델 목록 결정
    if args.model == "all":
        models_to_run = list(MODEL_CONFIGS.keys())
    else:
        models_to_run = [args.model]

    seeds = SEEDS[: args.repeats]
    total_runs = len(models_to_run) * len(seeds)

    print("=" * 60)
    print(f"  YOLO 반복 실험 자동화")
    print(f"  모델: {models_to_run}")
    print(f"  반복: {len(seeds)}회 (seeds: {seeds})")
    print(f"  총 실험 수: {total_runs}")
    print("=" * 60)

    results = []
    run_count = 0

    for model_key in models_to_run:
        model_cfg = MODEL_CONFIGS[model_key]

        for i, seed in enumerate(seeds):
            run_count += 1
            run_label = f"[{run_count}/{total_runs}] {model_cfg.project_name} | seed={seed}"
            cmd = build_command(model_cfg, seed)

            print("\n" + "=" * 60)
            print(f"  {run_label}")
            print("=" * 60)

            if args.dry_run:
                print(f"  weights        : {model_cfg.weights}")
                print(f"  project        : ./runs/{model_cfg.project_name}")
                print(f"  name           : {model_cfg.experiment_name}_seed{seed}")
                print(f"  seed           : {seed}")
                print(f"  epochs         : {COMMON_PARAMS['epochs']}")
                print(f"  batch          : {COMMON_PARAMS['batch']}")
                print(f"  lr0            : {COMMON_PARAMS['lr0']}")
                print(f"  command        : {' '.join(cmd)}")
                results.append((run_label, "DRY_RUN"))
                continue

            start_time = time.time()
            try:
                proc = subprocess.run(cmd, check=True)
                elapsed = time.time() - start_time
                status = f"SUCCESS ({elapsed / 60:.1f}min)"
                print(f"\n  {run_label} — {status}")
            except subprocess.CalledProcessError as e:
                elapsed = time.time() - start_time
                status = f"FAILED ({elapsed / 60:.1f}min) — exit code {e.returncode}"
                print(f"\n  {run_label} — {status}")
            except Exception as e:
                elapsed = time.time() - start_time
                status = f"FAILED ({elapsed / 60:.1f}min) — {e}"
                print(f"\n  {run_label} — {status}")
                traceback.print_exc()

            results.append((run_label, status))

    # 최종 요약
    print("\n" + "=" * 60)
    print("  실험 결과 요약")
    print("=" * 60)
    for label, status in results:
        print(f"  {label}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
