# Given a folder containing folders of generated images, this script will
# calculate scores for each image and take the average, max and min scores
import csv
import os

import cv2
import gc
import numpy as np
from natsort import natsorted
import torch
#torch.set_num_threads(1)
#torch.set_num_interop_threads(1)
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

from ..config import OUTPUT_FOLDER_NAME
from ..utils.argparser import parse_validation_args
from ..utils.validation import compute_ssim, compute_psnr, compute_mse, compute_cpl
from ..utils.validation import get_segmentation_model, calculate_semantic_segmentation_score

args = parse_validation_args()
carla_mode = True
show_messages = False

output_dir = os.path.join("../",OUTPUT_FOLDER_NAME,args.output_dir)

if not os.path.exists(output_dir):
    print(f"Output folder '{output_dir}' does not exist.")
    exit(1)

sim_folders = [f.path for f in os.scandir(output_dir) if f.is_dir() and f.name not in ["seg", "canny", "depth", "carla", "old"]]
csv_filename = os.path.join(output_dir, "results.csv")

real_folder = os.path.join(output_dir, "carla")
seg_folder = os.path.join(output_dir, "seg")

real_images = [f for f in natsorted(os.listdir(real_folder)) if f.lower().endswith(".png")]
seg_images = [f for f in natsorted(os.listdir(seg_folder)) if f.lower().endswith(".png")]

# Load the VGG16 model and create the transformation only once
if args.cpl:
    vgg16 = models.vgg16(pretrained=True).features.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # Converts a PIL image or numpy.ndarray in[255] to and rearranges dimensions
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

global_results = []

seg_model, seg_image_extractor = get_segmentation_model()

for simulated_folder in tqdm(natsorted(sim_folders), desc="Processing Folders"):
    local_csv_filename = os.path.join(simulated_folder,"results.csv")
    if show_messages:
        print(f"\nProcessing simulated images from folder: {simulated_folder}")
    sim_images = [f for f in natsorted(os.listdir(simulated_folder)) if f.lower().endswith(".png")]
    if len(sim_images) != len(real_images):
        print(f"Warning: The number of images in the simulated folder '{simulated_folder}' and real folder do not match. "
              f"Pairing up to the minimum count.")
    num_pairs = min(len(sim_images), len(real_images))

    # Lists to accumulate metrics for all image pairs
    ssim_scores = []
    psnr_scores = []
    mse_scores = []
    cpl_scores = []
    semantic_seg_scores = []
    results = []  # List to accumulate CSV rows

    for i in tqdm(range(num_pairs), desc=f"Frames for {simulated_folder}"):
        sim_path = os.path.join(simulated_folder, sim_images[i])
        real_path = os.path.join(real_folder, real_images[i])
        seg_path = os.path.join(seg_folder, seg_images[i])

        # Read images using cv2 (which loads as BGR)
        sim_image = cv2.imread(sim_path)
        real_image = cv2.imread(real_path)
        seg_image = cv2.imread(seg_path)
        if not os.path.exists(sim_path):
            print(f"{sim_path} does not exist!")
        if sim_image is None:
            print(f"Could not load simulated image: {sim_path}. Skipping.")
            continue
        if real_image is None:
            print(f"Could not load real image: {real_path}. Skipping.")
            continue

        # Resize simulated image to match real image dimensions if necessary

        real_image_resized = cv2.resize(real_image, (sim_image.shape[1], sim_image.shape[0]))
        seg_image_resized = cv2.resize(seg_image, (sim_image.shape[1], sim_image.shape[0]))

        # Compute metrics using the BGR images for cv2-based functions
        ssim_score = -1 if args.segmentation else compute_ssim(sim_image, real_image_resized)
        psnr_score = -1 if args.segmentation else compute_psnr(sim_image, real_image_resized)
        mse_score = -1 if args.segmentation else compute_mse(sim_image, real_image_resized)
        semantic_seg_score, _ = (-1,0) if not args.segmentation else calculate_semantic_segmentation_score(seg_model, seg_image_extractor, seg_image_resized, sim_image, carla_mode=carla_mode)
        #semantic_seg_score = 0.0

        # For CPL, convert images to RGB
        if args.cpl:
            sim_image_rgb = cv2.cvtColor(sim_image, cv2.COLOR_BGR2RGB)
            real_image_rgb = cv2.cvtColor(real_image_resized, cv2.COLOR_BGR2RGB)
            cpl_score = compute_cpl(sim_image_rgb, real_image_rgb, vgg16, transform)
        else:
            cpl_score = -1


        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
        mse_scores.append(mse_score)
        cpl_scores.append(cpl_score)
        semantic_seg_scores.append(semantic_seg_score)

        # Append the result row for this pair
        results.append([os.path.basename(sim_path), os.path.basename(real_path),
                        ssim_score, psnr_score, mse_score, cpl_score, semantic_seg_score])

        if show_messages:
            print(f"Pair {i + 1}: {os.path.basename(sim_path)} vs {os.path.basename(real_path)}")
            print(
                f"  SSIM: {ssim_score:.4f}, PSNR: {psnr_score:.4f}, MSE: {mse_score:.4f}, CPL: {cpl_score:.4f}, SSS: {semantic_seg_score:.4f}")

        sim_image, real_image, seg_image, real_image_resized, seg_image_resized = None, None, None, None, None
        gc.collect()
        #torch.cuda.empty_cache()

    # Append average metrics as the final row
    if ssim_scores:
        averages = ["AVERAGE", simulated_folder,
                    np.mean(ssim_scores),
                    np.mean(psnr_scores),
                    np.mean(mse_scores),
                    np.mean(cpl_scores),
                    np.mean(semantic_seg_scores)]
        mins = ["MIN", simulated_folder,
                    np.min(ssim_scores),
                    np.min(psnr_scores),
                    np.min(mse_scores),
                    np.min(cpl_scores),
                    np.min(semantic_seg_scores)]
        maxs = ["MAX", simulated_folder,
                    np.max(ssim_scores),
                    np.max(psnr_scores),
                    np.max(mse_scores),
                    np.max(cpl_scores),
                    np.max(semantic_seg_scores)]
        results.append(averages)
        results.append(mins)
        results.append(maxs)
        global_results.append(averages)
        global_results.append(mins)
        global_results.append(maxs)

    # Write results to CSV file
    with open(local_csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Simulated Image", "Real Image", "SSIM", "PSNR", "MSE", "CPL", "SSS"])
        writer.writerows(results)

    print(f"\nResults saved to {local_csv_filename}")

    if ssim_scores:
        print(f"\nAverage metrics over all pairs for {simulated_folder}:")
        print(f"  SSIM: {np.mean(ssim_scores):.4f}")
        print(f"  PSNR: {np.mean(psnr_scores):.4f}")
        print(f"  MSE: {np.mean(mse_scores):.4f}")
        print(f"  CPL: {np.mean(cpl_scores):.4f}")
        print(f"  SSS: {np.mean(semantic_seg_scores):.4f}")

if len(sim_folders) > 1:
    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Type", "Folder-Name", "SSIM", "PSNR", "MSE", "CPL", "SSS"])
        writer.writerows(global_results)

    print(f"\nGlobal Results saved to {csv_filename}")