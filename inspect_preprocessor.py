import argparse
import json
from transformers import AutoImageProcessor


def main():
    parser = argparse.ArgumentParser(description="Inspect a HuggingFace image processor.")
    parser.add_argument("--model_name", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--image_path", default=None, help="Optional image path to run preprocessing on")
    args = parser.parse_args()

    print(f"Loading processor: {args.model_name}\n")
    processor = AutoImageProcessor.from_pretrained(args.model_name)

    cfg = processor.to_dict()
    print(json.dumps(cfg, indent=2, default=str))

    if args.image_path:
        from PIL import Image
        image = Image.open(args.image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        print(f"\n[Processed Tensor]")
        print(f"  shape     : {tuple(pixel_values.shape)}")
        print(f"  min value : {pixel_values.min().item():.4f}")
        print(f"  max value : {pixel_values.max().item():.4f}")
        print(f"  dtype     : {pixel_values.dtype}")


if __name__ == "__main__":
    main()
