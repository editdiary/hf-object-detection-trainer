"""
Object Detection 모델 → ONNX (FP16) 변환 스크립트

사용법:
    # HuggingFace 모델 변환
    python export_onnx.py real_runs/huggingface/real_detr_r50/repeat-exp_seed42_1/best_model

    # torchvision Faster R-CNN (.pth) 변환
    python export_onnx.py --frcnn real_runs/frcnn_r50fpnv2/repeat-exp_seed42_1/best_model/model.pth

    # 전체 HF 모델 일괄 변환
    python export_onnx.py --all

    # 특정 모델 타입만
    python export_onnx.py --all --filter rtdetr

    # FP32 ONNX도 함께 저장
    python export_onnx.py --all --keep-fp32
"""

import argparse
import sys
import warnings
from pathlib import Path

import shutil

import numpy as np
import onnx
import torch
from onnx import TensorProto, numpy_helper
from onnxconverter_common import float16
from transformers import AutoModelForObjectDetection, AutoImageProcessor


def convert_to_fp16(onnx_path: Path):
    """ONNX 모델을 FP16으로 변환 (onnxconverter_common + Cast 노드 수정)."""
    model = onnx.load(str(onnx_path))

    # onnxconverter_common의 remove_unnecessary_cast_node가
    # subgraph(If/Loop 등)를 가진 모델에서 crash하는 버그 우회
    original_fn = float16.remove_unnecessary_cast_node
    def _safe_remove_cast(graph):
        try:
            original_fn(graph)
        except AttributeError:
            pass  # subgraph 처리 실패 시 cleanup 단계 건너뛰기
    float16.remove_unnecessary_cast_node = _safe_remove_cast

    try:
        model_fp16 = float16.convert_float_to_float16(
            model,
            disable_shape_infer=False,
            keep_io_types=False,
        )
    finally:
        float16.remove_unnecessary_cast_node = original_fn

    # onnxconverter_common이 원본 Cast 노드의 target type을 변경하지 않으므로 후처리
    # 단, onnxconverter가 삽입한 보호 Cast (_input_cast, _output_cast)는 유지
    # subgraph (If/Loop 내부)도 재귀적으로 처리
    def _fix_cast_nodes(graph):
        count = 0
        for node in graph.node:
            # subgraph 재귀 처리
            for attr in node.attribute:
                if attr.g and attr.g.node:
                    count += _fix_cast_nodes(attr.g)
            if node.op_type == "Cast":
                if "_cast" in node.name:
                    continue
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == TensorProto.FLOAT:
                        attr.i = TensorProto.FLOAT16
                        count += 1
        return count

    cast_fixed = _fix_cast_nodes(model_fp16.graph)
    if cast_fixed > 0:
        print(f"    Fixed {cast_fixed} Cast nodes (FP32 → FP16)")

    onnx.save(model_fp16, str(onnx_path))
    return model_fp16


def export_to_onnx_fp16(model_dir: str, output_path: str | None = None, opset: int = 17, keep_fp32: bool = False):
    model_dir = Path(model_dir)
    if output_path is None:
        output_path = model_dir / "model_fp16.onnx"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading model from {model_dir} ...")
    model = AutoModelForObjectDetection.from_pretrained(model_dir)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(model_dir)

    # 입력 사이즈 결정 (preprocessor_config에서)
    h = getattr(processor, "size", {}).get("height", 640)
    w = getattr(processor, "size", {}).get("width", 640)
    # DETR 계열은 size가 dict가 아닐 수 있음
    if isinstance(getattr(processor, "size", None), dict):
        size_dict = processor.size
        if "shortest_edge" in size_dict:
            h = w = size_dict["shortest_edge"]
        elif "height" in size_dict and "width" in size_dict:
            h, w = size_dict["height"], size_dict["width"]
    print(f"    Input size: {h}x{w}")

    dummy_input = torch.randn(1, 3, h, w)

    print(f"[2/4] Exporting to ONNX (FP32, opset={opset}) ...")
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", message=".*Exporting aten::index.*")
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            opset_version=opset,
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "logits": {0: "batch_size"},
                "pred_boxes": {0: "batch_size"},
            },
        )

    if keep_fp32:
        fp32_path = output_path.with_name(output_path.name.replace("_fp16", "_fp32"))
        if fp32_path == output_path:
            fp32_path = output_path.with_suffix(".fp32.onnx")
        shutil.copy2(output_path, fp32_path)
        fp32_size = fp32_path.stat().st_size / (1024 * 1024)
        print(f"    FP32 saved: {fp32_path} ({fp32_size:.1f} MB)")

    print(f"[3/4] Converting to FP16 ...")
    model_fp16 = convert_to_fp16(output_path)

    print(f"[4/4] Validating ...")
    onnx.checker.check_model(model_fp16)

    fp16_size = output_path.stat().st_size / (1024 * 1024)
    print(f"    Size: {fp16_size:.1f} MB")
    print(f"    Saved: {output_path}")
    return output_path


def export_frcnn_to_onnx_fp16(
    model_path: str,
    output_path: str | None = None,
    opset: int = 17,
    num_classes: int | None = None,
    image_size: int = 640,
    keep_fp32: bool = False,
):
    """torchvision Faster R-CNN (.pth) → ONNX FP16 변환."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model_path = Path(model_path)
    if output_path is None:
        output_path = model_path.with_name("model_fp16.onnx")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # num_classes 자동 감지 (state_dict에서)
    print(f"[1/4] Loading Faster R-CNN from {model_path} ...")
    state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
    if num_classes is None:
        num_classes = state_dict["roi_heads.box_predictor.cls_score.weight"].shape[0]
        print(f"    Auto-detected num_classes (incl. background): {num_classes}")

    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.transform.min_size = (image_size,)
    model.transform.max_size = image_size
    model.load_state_dict(state_dict)
    model.eval()
    print(f"    Input size: {image_size}x{image_size}")

    dummy_input = torch.randn(1, 3, image_size, image_size)

    print(f"[2/4] Exporting to ONNX (FP32, opset={opset}) ...")
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", message=".*Exporting aten::index.*")
        warnings.filterwarnings("ignore", message=".*ONNX export of operator.*")
        warnings.filterwarnings("ignore", message=".*sourceTensor.detach.*")
        torch.onnx.export(
            model,
            ([dummy_input[0]],),  # Faster R-CNN expects list of tensors
            str(output_path),
            opset_version=opset,
            input_names=["image"],
            output_names=["boxes", "labels", "scores"],
            dynamic_axes={
                "image": {1: "height", 2: "width"},
                "boxes": {0: "num_detections"},
                "labels": {0: "num_detections"},
                "scores": {0: "num_detections"},
            },
        )

    if keep_fp32:
        fp32_path = output_path.with_name(output_path.name.replace("_fp16", "_fp32"))
        if fp32_path == output_path:
            fp32_path = output_path.with_suffix(".fp32.onnx")
        shutil.copy2(output_path, fp32_path)
        fp32_size = fp32_path.stat().st_size / (1024 * 1024)
        print(f"    FP32 saved: {fp32_path} ({fp32_size:.1f} MB)")

    print(f"[3/4] Converting to FP16 ...")
    model_fp16 = convert_to_fp16(output_path)

    print(f"[4/4] Validating ...")
    onnx.checker.check_model(model_fp16)

    fp16_size = output_path.stat().st_size / (1024 * 1024)
    print(f"    Size: {fp16_size:.1f} MB")
    print(f"    Saved: {output_path}")
    return output_path


def find_all_models(base_dir: str, filter_name: str | None = None):
    """real_runs/huggingface 아래의 모든 best_model 디렉토리를 찾음."""
    base = Path(base_dir)
    models = sorted(base.rglob("best_model/config.json"))
    result = []
    for cfg in models:
        model_dir = cfg.parent
        if filter_name and filter_name.lower() not in str(model_dir).lower():
            continue
        result.append(model_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description="Export object detection models to ONNX FP16")
    parser.add_argument("model_dir", nargs="?", help="Path to best_model directory (HF)")
    parser.add_argument("--frcnn", type=str, default=None,
                        help="Path to Faster R-CNN .pth weights file")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Number of classes incl. background (auto-detected if omitted)")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Input image size for Faster R-CNN (default: 640)")
    parser.add_argument("--all", action="store_true", help="Convert all models under real_runs/huggingface")
    parser.add_argument("--filter", type=str, default=None, help="Filter model dirs by name (e.g., 'rtdetr')")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--keep-fp32", action="store_true", help="Also save the FP32 ONNX model")
    args = parser.parse_args()

    if args.frcnn:
        output = None
        if args.output_dir:
            name = Path(args.frcnn).stem
            output = str(Path(args.output_dir) / f"{name}_fp16.onnx")
        export_frcnn_to_onnx_fp16(
            args.frcnn, output, opset=args.opset,
            num_classes=args.num_classes, image_size=args.image_size,
            keep_fp32=args.keep_fp32,
        )
    elif args.all:
        base = Path(__file__).parent / "real_runs" / "huggingface"
        models = find_all_models(str(base), args.filter)
        if not models:
            print("No models found.")
            sys.exit(1)
        print(f"Found {len(models)} model(s):\n")
        for m in models:
            print(f"  {m}")
        print()
        for m in models:
            print(f"{'='*60}")
            try:
                export_to_onnx_fp16(str(m), opset=args.opset, keep_fp32=args.keep_fp32)
            except Exception as e:
                print(f"  ERROR: {e}")
            print()
    elif args.model_dir:
        output = None
        if args.output_dir:
            name = Path(args.model_dir).parent.name + "_" + Path(args.model_dir).name
            output = str(Path(args.output_dir) / f"{name}_fp16.onnx")
        export_to_onnx_fp16(args.model_dir, output, opset=args.opset, keep_fp32=args.keep_fp32)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
