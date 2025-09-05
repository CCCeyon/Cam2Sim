import os
import re

from natsort import natsorted
import numpy as np
import torch
from tqdm import tqdm
import cv2
from PIL import Image
from datasets import Features, Value, Image as DSImage
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor, SegformerImageProcessor, \
    SegformerForSemanticSegmentation
from transformers import pipeline

from config import CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD, \
    LLAVA_MODEL, LLAVA_BATCH_SIZE, LLAVA_DESCRIPTION_PROMPT, CARLA_MODE, SEGFORMER_MODEL
from utils.stable_diffusion import get_device

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_video_dataset(dataset_path, video_filename, create_canny, face_blur, override = False):
    if not override:
        video = cv2.VideoCapture(video_filename)
    new_dataset = []

    images_folder = os.path.join(dataset_path,"images")
    canny_folder = os.path.join(dataset_path,"canny")

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)
    #if create_canny:
    #    os.makedirs(canny_folder, exist_ok=True)

    existing_frames = len([f for f in os.listdir(images_folder) if f.endswith(".png")])

    if not override:
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        max_digits = 5
    else:
        frame_count = existing_frames
        max_digits = 6


    previous = None

    if existing_frames == frame_count:
        print(f"⚠️ The Dataset-Folder contains the exact amount of frames ({existing_frames}) as the video ({frame_count}). Skipping frame extraction.")
        if override:
            for f in natsorted(os.listdir(images_folder)):
                if f.endswith(".png"):
                    entry = {
                    "image": os.path.join(images_folder, f),
                    "previous": previous if previous is not None else os.path.join(images_folder, f)
                }
                previous = entry["image"]
                new_dataset.append(entry)
        else:

            for idx in range(frame_count):
                entry = {
                    "image": os.path.join(images_folder, f"frame_{idx:0{max_digits}d}.png"),
                    "previous": previous if previous is not None else os.path.join(images_folder, f"frame_{idx:0{max_digits}d}.png")
                }
                previous = entry["image"]
                #if create_canny:
                #    entry["canny"] = os.path.join(canny_folder, f"frame_{idx:05d}.png")

                new_dataset.append(entry)
    else:
        with tqdm(total=frame_count, desc="Fetching Frames, Creating Canny" if create_canny else "Fetching Frames") as pbar:
            frame_id = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break

                if face_blur:
                    blur_faces_in_image(frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frame_path = os.path.join(images_folder, f"frame_{frame_id:05d}.png")
                pil_image.save(frame_path)

                entry = {
                    "image": frame_path,
                    "previous": previous if previous is not None else frame_path
                }
                previous = entry["image"]

                #if create_canny:
                #    canny_image = cv2.Canny(frame_rgb, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
                #    canny_image_pil = Image.fromarray(canny_image)
                #    canny_path = os.path.join(canny_folder, f"frame_{frame_id:05d}.png")
                #    canny_image_pil.save(canny_path)
                #    entry["canny"] = canny_path

                new_dataset.append(entry)
                frame_id += 1
                pbar.update(1)
    if not override:
        video.release()
    return new_dataset

def blur_faces_in_image(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Gesicht ausschneiden und blurren
        face_roi = frame[y:y + h, x:x + w]
        face_blur = cv2.GaussianBlur(face_roi, (10, 10), 30)
        frame[y:y + h, x:x + w] = face_blur

def describe_images(current_dataset):
    llava_device = get_device()
    print(f"Using Llava model on device: {llava_device}")
    llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        LLAVA_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(llava_device)

    llava_processor = AutoProcessor.from_pretrained(LLAVA_MODEL, use_fast=True)

    for i in tqdm(range(0, len(current_dataset), LLAVA_BATCH_SIZE), desc="Describing Images with LLaVA"):
        batch = current_dataset[i:i + LLAVA_BATCH_SIZE]

        image_paths = [item["image"] for item in batch]
        images = [Image.open(p).convert("RGB") for p in image_paths]

        llava_conversations = [
            [{
                "role": "user",
                "content": [
                    {"type": "text", "text": LLAVA_DESCRIPTION_PROMPT},
                    {"type": "image"},
                ],
            }]
            for _ in images
        ]

        llava_prompts = [
            llava_processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in llava_conversations
        ]

        inputs = llava_processor(
            images=images,
            text=llava_prompts,
            return_tensors='pt',
        ).to(llava_device, torch.float16)

        prompt_token_count = inputs['input_ids'].shape[-1]

        outputs = llava_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=llava_processor.tokenizer.eos_token_id
        )

        for item, output in zip(batch, outputs):
            caption = llava_processor.decode(output[prompt_token_count:], skip_special_tokens=True)
            item["caption"] = caption
        pass

def get_dataset_features(create_canny):
    if create_canny:
        return Features({
            "image": DSImage(),
            "canny": DSImage(),
            "segmentation": DSImage(),
            "depth": DSImage(),
            "previous": DSImage(),
            "caption": Value("string")
        })
    else:
        return Features({
            "image": DSImage(),
            "segmentation": DSImage(),
            "depth": DSImage(),
            "previous": DSImage(),
            "caption": Value("string")
        })

def decode_cityscapes_mask(mask):
    segmentation_colors = [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
        (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
        (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
        (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]
    if CARLA_MODE:
        segmentation_colors = [
            (128, 64, 128), (128, 64, 128),
            (0,0,0), (0,0,0), (0,0,0), (0,0,0),
            (250, 170, 30), (220, 220, 0),
            (0,0,0), (0,0,0), (0,0,0),
            (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
            (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
        ]

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(segmentation_colors):
        color_mask[mask == class_id] = color
    return Image.fromarray(color_mask)

def add_existing_data(current_dataset, dataset_path, image_type):
    folder = os.path.join(dataset_path, image_type)
    os.makedirs(folder, exist_ok=True)

    existing_depth = len([f for f in os.listdir(folder) if f.endswith(".png")])

    if existing_depth == len(current_dataset):
        print(f"The Dataset-Folder already contains {existing_depth} {folder} images. Skipping {folder}-image creation.")
        for entry in current_dataset:
            image_path = entry["image"]
            image_name = os.path.basename(image_path)
            entry[image_type] = os.path.join(folder, image_name)
        return True
    return False

def create_depth_data(current_dataset, dataset_path):
    depth_folder = os.path.join(dataset_path, "depth")
    if add_existing_data(current_dataset, dataset_path, "depth"):
        return
    

    device = get_device()

    depth_estimator = pipeline('depth-estimation', model="Intel/zoedepth-nyu-kitti")
    for entry in tqdm(current_dataset, desc="Creating Depth-Image Estimation for each frame"):
        image_path = entry["image"]

        seg_image = Image.open(entry["segmentation"]).convert("RGB")
        seg_image = np.array(seg_image)

        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")

        depth_image = depth_estimator(image)['depth']
        depth_image = np.array(depth_image)
        depth_image = depth_image[:, :, None]
        depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)

        if depth_image.ndim == 2:
            depth_image = np.stack([depth_image]*3, axis=2)

        # Alle schwarzen Pixel im seg_image finden
        mask = np.all(seg_image == [0, 0, 0], axis=-1)  # Shape: (H, W)

        # Schwarze Bereiche im seg_image → weiß im depth_image
        depth_image[mask] = [255, 255, 255]

        depth_image = Image.fromarray(depth_image)

        depth_path = os.path.join(depth_folder, image_name)
        depth_image.save(depth_path)

        entry["depth"] = depth_path

def create_canny_data(current_dataset, dataset_path):
    canny_folder = os.path.join(dataset_path, "canny")
    os.makedirs(canny_folder, exist_ok=True)

    if add_existing_data(current_dataset, dataset_path, "canny"):
        return
    
    for entry in tqdm(current_dataset, desc="Creating Canny-Images for each frame"):
        image_path = entry["image"]

        seg_image = Image.open(entry["segmentation"]).convert("RGB")
        seg_image = np.array(seg_image)

        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")

        mask = np.all(seg_image == [0, 0, 0], axis=-1)  # Shape: (H, W)
        
        image_np = np.array(image)
        canny_image = cv2.Canny(image_np, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        canny_image[mask] = 0
        canny_image = Image.fromarray(canny_image)

        canny_path = os.path.join(canny_folder, image_name)
        canny_image.save(canny_path)

        entry["canny"] = canny_path
        
    

def create_segmentation_data(current_dataset, dataset_path):
    segmentation_folder = os.path.join(dataset_path, "segmentation")
    
    if add_existing_data(current_dataset, dataset_path, "segmentation"):
        return

    device = get_device()

    image_extractor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
    model.eval().to(device)

    for entry in tqdm(current_dataset, desc="Creating Segmentation-Map for each frame"):
        image_path = entry["image"]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")

        inputs = image_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [B, num_classes, H, W]
            upsampled = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
            predicted = upsampled.argmax(1)[0].cpu().numpy()

        segmentation_path = os.path.join(segmentation_folder, image_name)
        segmentation_image = decode_cityscapes_mask(predicted)
        segmentation_image.save(segmentation_path)

        entry["segmentation"] = segmentation_path
