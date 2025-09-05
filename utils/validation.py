import os

import cv2
import numpy as np
from natsort import natsorted
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from config import SEGFORMER_MODEL
from utils.dataset import decode_cityscapes_mask
from utils.stable_diffusion import get_device


# The following functions were copied from a different private repository

def compute_ssim(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(image1_gray, image2_gray, full=True)
    return score

def compute_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compute_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def compute_cpl(image1, image2, model, transform):
    import torch
    import torch.nn.functional
# image1 and image2 should be in RGB order (numpy arrays)
    img1_pil = Image.fromarray(image1)
    img2_pil = Image.fromarray(image2)
    img1_tensor = transform(img1_pil).unsqueeze(0)
    img2_tensor = transform(img2_pil).unsqueeze(0)
    with torch.no_grad():
        features1 = model(img1_tensor)
        features2 = model(img2_tensor)

    return torch.nn.functional.mse_loss(features1, features2).item()



def get_segmentation_model():
    device = get_device()
    print(device)
    image_extractor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
    model.eval().to(device)

    return model, image_extractor

def encode_cityscapes_mask(rgb_img, carla_mode=False):
    import torch
    rgb = np.array(rgb_img)
    h, w, _ = rgb.shape
    label_mask = np.full((h, w), fill_value=-1, dtype=np.int64)

    if carla_mode:
        segmentation_colors = [
            (128, 64, 128), (128, 64, 128),
            (0,0,0), (0,0,0), (0,0,0), (0,0,0),
            (250, 170, 30), (220, 220, 0),
            (0,0,0), (0,0,0), (0,0,0),
            (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
            (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
        ]
    else:
        segmentation_colors = [
            (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
            (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
            (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
            (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
        ]

    # Mappe Farben -> IDs
    for class_id, color in enumerate(segmentation_colors):
        mask = np.all(rgb == color, axis=-1)
        label_mask[mask] = class_id

    return torch.from_numpy(label_mask)


def calculate_semantic_segmentation_score(model, image_extractor, image_seg, image_created, carla_mode=False):
    import torch
    inputs = image_extractor(images=image_created, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [B, num_classes, H, W]
        target_size = image_created.shape[:2]  # cv2: (H, W)

        upsampled = torch.nn.functional.interpolate(
            logits, size=target_size, mode="bilinear", align_corners=False
        )

        predicted = upsampled.argmax(1)[0].cpu().numpy()  # (H, W)

    del outputs, logits, upsampled

    # Farbmaske generieren (nur für Visualisierung, optional)
    segmentation_image = decode_cityscapes_mask(predicted)

    # IDs für Vergleich erzeugen (2D!)
    if image_seg is None:
        return -1, segmentation_image

    pred_ids = encode_cityscapes_mask(segmentation_image, carla_mode=carla_mode)  # (H, W)
    gt_ids   = encode_cityscapes_mask(image_seg, carla_mode=carla_mode)           # (H, W)

    if isinstance(pred_ids, np.ndarray):
        pred_ids = torch.from_numpy(pred_ids)
    if isinstance(gt_ids, np.ndarray):
        gt_ids = torch.from_numpy(gt_ids)

    pred_ids = pred_ids.float()
    gt_ids   = gt_ids.float()

    mse_score = torch.nn.functional.mse_loss(pred_ids.float(), gt_ids.float())
    #print("MSE auf class IDs:", mse_score.item())
    return mse_score.item(), segmentation_image