"""
반복 실험 자동화 스크립트.

사용법:
    python run_experiments.py                # 전체 실험 (모든 모델 × 5회)
    python run_experiments.py --model r34    # RT-DETR v2 R34만 5회
    python run_experiments.py --model r50    # RT-DETR v2 R50만 5회
    python run_experiments.py --model detr   # DETR만 5회
    python run_experiments.py --repeats 3    # 반복 횟수 변경
    python run_experiments.py --dry-run      # 실제 학습 없이 설정만 출력
"""

import argparse
import time
import traceback
from dataclasses import dataclass, field

# =========================================================
# 1. 실험 설정 정의
# =========================================================

# 반복 실험마다 사용할 시드 목록 (재현 가능한 고정 시드)
SEEDS = [42, 42, 42, 42, 42]

# 모든 모델에 공통으로 적용할 학습 파라미터 (필요시 수정)
COMMON_PARAMS = {
    "BATCH_SIZE": 16,
    "EPOCHS": 150,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "OPTIM": "adamw_torch",
    "LR_SCHEDULER_TYPE": "cosine",
    "MAX_GRAD_NORM": 0.1,
    "FREEZE_BACKBONE": False,
    "USE_EARLY_STOPPING": True,
    "EARLY_STOP_PATIENCE": 40,
    "WEIGHT_LOSS_VFL": 1.0,
    "WEIGHT_LOSS_BBOX": 5.0,
    "WEIGHT_LOSS_GIOU": 2.0,
    "FOCAL_LOSS_ALPHA": 0.75,
    "FOCAL_LOSS_GAMMA": 2.0,
    "MATCHER_CLASS_COST": 2.0,
    "MATCHER_BBOX_COST": 0.5,
    "MATCHER_GIOU_COST": 0.5,
}


@dataclass
class ModelConfig:
    """모델별 실험 설정."""
    key: str                    # 커맨드라인에서 사용할 짧은 이름
    checkpoint: str             # HuggingFace 모델 체크포인트
    project_name: str           # runs/{project_name}/ 하위에 결과 저장
    experiment_name: str        # 실험 폴더 접두사
    # Loss 하이퍼파라미터 (모델 아키텍처에 따라 다름)
    loss_params: dict = field(default_factory=dict)
    # 모델별로 공통 파라미터를 오버라이드할 항목
    overrides: dict = field(default_factory=dict)


# --- 모델별 설정 ---
MODEL_CONFIGS = {
    # "v1_r34": ModelConfig(
    #     key="v1_r18",
    #     checkpoint="PekingU/rtdetr_r18vd",
    #     project_name="real_rtdetr_v1_r18",
    #     experiment_name="repeat-exp",
    # ),
    "r18": ModelConfig(
        key="r18",
        checkpoint="PekingU/rtdetr_v2_r18vd",
        project_name="real_rtdetr_v2_r18",
        experiment_name="repeat-exp",
    ),
    "r34": ModelConfig(
        key="r34",
        checkpoint="PekingU/rtdetr_v2_r34vd",
        project_name="real_rtdetr_v2_r34",
        experiment_name="repeat-exp",
    ),
    "r50": ModelConfig(
        key="r50",
        checkpoint="PekingU/rtdetr_v2_r50vd",
        project_name="real_rtdetr_v2_r50",
        experiment_name="repeat-exp",
    ),
    "r101": ModelConfig(
        key="r101",
        checkpoint="PekingU/rtdetr_v2_r101vd",
        project_name="real_rtdetr_v2_r101",
        experiment_name="repeat-exp",
    ),
    "detr": ModelConfig(
        key="detr",
        checkpoint="facebook/detr-resnet-50",
        project_name="real_detr_r50",
        experiment_name="repeat-exp",
        overrides={
            "BATCH_SIZE": 8,        # DETR은 메모리 사용량이 크므로 배치 축소
            "IMAGE_SIZE": 800,      # DETR 기본 입력 크기 800x800
        },
    ),
}


# =========================================================
# 2. Config 클래스 동적 패치 & 학습 실행
# =========================================================

def apply_config(model_cfg: ModelConfig, seed: int, run_idx: int):
    """Config 클래스의 속성을 실험 설정에 맞게 동적으로 변경."""
    from src.config import Config

    # 공통 파라미터 적용
    for key, value in COMMON_PARAMS.items():
        setattr(Config, key, value)

    # 모델별 오버라이드 적용
    for key, value in model_cfg.overrides.items():
        setattr(Config, key, value)

    # # 모델별 Loss 파라미터 적용
    # for key, value in model_cfg.loss_params.items():
    #     setattr(Config, key, value)

    # 모델 체크포인트 & 프로젝트 설정
    Config.MODEL_CHECKPOINT = model_cfg.checkpoint
    Config.PROJECT_NAME = model_cfg.project_name
    Config.EXPERIMENT_NAME = f"{model_cfg.experiment_name}_seed{seed}_"

    # 시드 설정
    Config.SEED = seed


def run_single_experiment(model_cfg: ModelConfig, seed: int, run_idx: int):
    """단일 실험 1회 실행."""
    import importlib
    import train as train_module

    # Config 패치
    apply_config(model_cfg, seed, run_idx)

    # train 모듈 재로드 (Config 변경사항 반영)
    importlib.reload(train_module)

    # 학습 실행
    train_module.main()


# =========================================================
# 3. 메인 실행 로직
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="반복 실험 자동화 스크립트")
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
        models_to_run = ["r18", "r34", "r50", "r101"]
    else:
        models_to_run = [args.model]

    seeds = SEEDS[: args.repeats]
    total_runs = len(models_to_run) * len(seeds)

    print("=" * 60)
    print(f"  반복 실험 자동화")
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

            print("\n" + "=" * 60)
            print(f"  🚀 {run_label}")
            print("=" * 60)

            if args.dry_run:
                apply_config(model_cfg, seed, i + 1)
                from src.config import Config
                print(f"  MODEL_CHECKPOINT : {Config.MODEL_CHECKPOINT}")
                print(f"  PROJECT_NAME     : {Config.PROJECT_NAME}")
                print(f"  EXPERIMENT_NAME  : {Config.EXPERIMENT_NAME}")
                print(f"  SEED             : {Config.SEED}")
                print(f"  BATCH_SIZE       : {Config.BATCH_SIZE}")
                print(f"  EPOCHS           : {Config.EPOCHS}")
                print(f"  LEARNING_RATE    : {Config.LEARNING_RATE}")
                results.append((run_label, "DRY_RUN"))
                continue

            start_time = time.time()
            try:
                run_single_experiment(model_cfg, seed, i + 1)
                elapsed = time.time() - start_time
                status = f"SUCCESS ({elapsed / 60:.1f}분)"
                print(f"\n  ✅ {run_label} — {status}")
            except Exception as e:
                elapsed = time.time() - start_time
                status = f"FAILED ({elapsed / 60:.1f}분) — {e}"
                print(f"\n  ❌ {run_label} — {status}")
                traceback.print_exc()

            results.append((run_label, status))

    # 최종 요약
    print("\n" + "=" * 60)
    print("  📊 실험 결과 요약")
    print("=" * 60)
    for label, status in results:
        print(f"  {label}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
