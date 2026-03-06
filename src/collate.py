"""
Custom collate function for variable-size object detection batches.
"""
import torch


def get_collate_fn(processor):
    """
    Returns a collate_fn that pads all images in a batch to the same spatial
    size and produces a pixel_mask (1 = real pixel, 0 = padding).
    """
    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]

        max_h = max(img.shape[1] for img in pixel_values)
        max_w = max(img.shape[2] for img in pixel_values)

        padded_pixel_values = []
        pixel_masks = []

        for img in pixel_values:
            c, h, w = img.shape

            padded_img = torch.zeros((c, max_h, max_w), dtype=img.dtype)
            padded_img[:, :h, :w] = img
            padded_pixel_values.append(padded_img)

            mask = torch.zeros((max_h, max_w), dtype=torch.long)
            mask[:h, :w] = 1
            pixel_masks.append(mask)

        return {
            "pixel_values": torch.stack(padded_pixel_values),
            "pixel_mask": torch.stack(pixel_masks),
            "labels": labels,
        }

    return collate_fn
