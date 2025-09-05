import csv
import os

import cv2
from natsort import natsorted
import numpy as np
from tqdm import tqdm

from config import OUTPUT_FOLDER_NAME
from utils.argparser import parse_validation_args
from utils.yolo import calculate_yolo_image, load_yolo_model
from shapely.geometry import box
from collections import Counter
from PIL import Image
import torchvision.transforms as transforms

args = parse_validation_args()

output_dir = os.path.join("../",OUTPUT_FOLDER_NAME,args.output_dir)
VEHICLE_CLASSES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck", 1: "bicycle"}

if not os.path.exists(output_dir):
    print(f"Output folder '{output_dir}' does not exist.")
    exit(1)

real_folder = os.path.join(output_dir, "carla")
real_images = [f for f in natsorted(os.listdir(real_folder)) if f.lower().endswith(".png")]

sim_folders = [f.path for f in os.scandir(output_dir) if f.is_dir() and f.name not in ["seg", "canny", "depth", "carla", "old"]]
yolo_model = load_yolo_model()

def iou(box1, box2):
    b1 = box(box1[0], box1[1], box1[2], box1[3])
    b2 = box(box2[0], box2[1], box2[2], box2[3])
    return b1.intersection(b2).area / b1.union(b2).area if b1.union(b2).area > 0 else 0


def extract_vehicles(results):
    vehicles = []
    for r in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = r
        cls = int(cls)
        if cls in VEHICLE_CLASSES:
            vehicles.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "class": VEHICLE_CLASSES[cls]
            })
    return vehicles


def match_vehicles(real_vehicles, gen_vehicles, iou_thresh=0.5):
    matches = []
    used_gen = set()

    for rv in real_vehicles:
        best_match = None
        best_iou = 0
        for i, gv in enumerate(gen_vehicles):
            if i in used_gen:
                continue
            iou_score = iou(rv["bbox"], gv["bbox"])
            if iou_score > best_iou:
                best_iou = iou_score
                best_match = (rv, gv, iou_score)

        if best_match and best_iou >= iou_thresh:
            matches.append(best_match)
            used_gen.add(gen_vehicles.index(best_match[1]))

    return matches, used_gen

def filter_large_vehicles(vehicles, min_area=200):
    return [v for v in vehicles if (v["bbox"][2]-v["bbox"][0])*(v["bbox"][3]-v["bbox"][1]) >= min_area]

def draw_filtered_vehicles_pil(pil_image, vehicles, color=(0,255,0), thickness=2):
    """
    Draw only the filtered vehicles on a PIL image.
    Returns an OpenCV image (numpy array) suitable for cv2.imshow.
    """
    img = np.array(pil_image.convert("RGB"))  # convert to numpy HWC
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV expects BGR

    for v in vehicles:
        x1, y1, x2, y2 = map(int, v['bbox'])
        cls_name = v['class']
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, cls_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
        ]
        )

csv_filename = os.path.join(output_dir,"veh_results.csv")
global_csv_rows = []

for simulated_folder in tqdm(natsorted(sim_folders), desc="Processing Folders"):

    #if "_rotate" not in simulated_folder:
     #   continue

    local_csv_filename = os.path.join(simulated_folder,"veh_results.csv")

    sim_images = [f for f in natsorted(os.listdir(simulated_folder)) if f.lower().endswith(".png")]
    if len(sim_images) != len(real_images):
        print(f"Warning: The number of images in the simulated folder '{simulated_folder}' and real folder do not match. "
              f"Pairing up to the minimum count.")
    num_pairs = min(len(sim_images), len(real_images))

    csv_rows = []
    veh_offset = []
    recalls = []
    precisions = []
    avg_ious = []
    class_accuracies = []
    mean_aps = []
    avg_confidences = []
    confidence_differences = []
    all_matched_gen_confidences = []

    for i in tqdm(range(num_pairs), desc=f"Frames for {simulated_folder}"):
        sim_path = os.path.join(simulated_folder, sim_images[i])
        real_path = os.path.join(real_folder, real_images[i])

        real_image = image_transforms(Image.open(real_path))
        sim_image = image_transforms(Image.open(sim_path))

        yolo_results_real, _ = calculate_yolo_image(yolo_model, real_image )
        yolo_results_sim, _ = calculate_yolo_image(yolo_model, sim_image )

        real_vehicles = extract_vehicles(yolo_results_real)
        gen_vehicles = extract_vehicles(yolo_results_sim)

        real_vehicles = filter_large_vehicles(real_vehicles, min_area=600)
        gen_vehicles = filter_large_vehicles(gen_vehicles, min_area=600)

        # IoU threshold based matching for recall, precision, extra/missing
        matches, used_gen = match_vehicles(real_vehicles, gen_vehicles)
        recall = len(matches) / len(real_vehicles) if real_vehicles else 0
        precision = len(matches) / len(gen_vehicles) if gen_vehicles else 0

        matched_gen_confidences = [rv["conf"] - gv["conf"] for rv, gv, _ in matches]
        all_matched_gen_confidences = all_matched_gen_confidences + matched_gen_confidences
        avg_confidence_difference = np.mean(matched_gen_confidences) if matched_gen_confidences else 0

        #real_confidences = [rv["conf"] for rv in real_vehicles]
        #avg_real_conf = np.mean(real_confidences) if real_confidences else 0
        #confidence_difference = avg_real_conf - avg_confidence


        # IoU stats and class agreement
        ious = [m[2] for m in matches]
        avg_iou = np.mean(ious) if ious else 0
        class_matches = sum(1 for rv, gv, _ in matches if rv["class"] == gv["class"])
        class_accuracy = class_matches / len(matches) if matches else 0

        extra = len(gen_vehicles) - len(used_gen)
        missing = len(real_vehicles) - len(matches)

        # ---------------------
        # AP Calculation across IoU thresholds (mimicking the paper)
        # ---------------------
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        ap_per_threshold = []

        for thr in iou_thresholds:
            m, u = match_vehicles(real_vehicles, gen_vehicles, iou_thresh=thr)
            ap = len(m)/len(gen_vehicles) if gen_vehicles else 0
            ap_per_threshold.append(ap)

        mean_ap = np.mean(ap_per_threshold)

        # ---------------------
        # Report
        # ---------------------
        if False:
            print("--- Vehicle Comparison Report ---")
            print(f"Real image vehicles: {len(real_vehicles)}")
            print(f"Generated image vehicles: {len(gen_vehicles)}")

            print("\nReal vehicle classes:", Counter([v["class"] for v in real_vehicles]))
            print("Generated vehicle classes:", Counter([v["class"] for v in gen_vehicles]))

            print("\nMatches found:", len(matches))
            print("Recall (coverage of real vehicles):", round(recall,3))
            print("Precision (how many generated match real):", round(precision,3))
            print("Average IoU (overlapping quality):", round(avg_iou,3))
            print("Class agreement on matches:", round(class_accuracy,3))
            print(f"Extra vehicles in generated: {extra}")
            print(f"Missing vehicles from generated: {missing}")

            print("\nAP at different IoU thresholds:", [round(a,3) for a in ap_per_threshold])
            print("Mean AP (mimicking paper):", round(mean_ap,3))

            print("Avg. confidence:", avg_confidence)
            print("confidence Difference:", confidence_difference)

        # Only plot the filtered real vehicles
        #vis_real_filtered = draw_filtered_vehicles_pil(real_image, real_vehicles)

        # Only plot generated vehicles that matched the real vehicles
        #vis_gen_filtered = draw_filtered_vehicles_pil(sim_image, [gv for rv, gv, _ in matches])

        csv_rows.append([
            sim_path,
            real_path,
            len(real_vehicles),
            len(gen_vehicles),
            Counter([v["class"] for v in real_vehicles]),
            Counter([v["class"] for v in gen_vehicles]),
            len(matches),
            round(recall,3),
            round(precision,3),
            round(avg_iou,3),
            round(class_accuracy,3),
            extra,
            missing,
            round(mean_ap,3),
            avg_confidence_difference,
        ])

        veh_offset.append(extra + missing)
        recalls.append(recall)
        precisions.append(precision)
        avg_ious.append(avg_iou)
        class_accuracies.append(class_accuracy)
        mean_aps.append(mean_ap)
        confidence_differences.append(avg_confidence_difference)

        #cv2.imshow("Real Large Vehicles", vis_real_filtered)
        #cv2.imshow("Generated Matched Vehicles", vis_gen_filtered)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    with open(local_csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Simulated Image", "Real Image", "Real Vehicles", "Generated Vehicles", "Real Classes", "Generated Classes", "Matches", "Recall", "Precision", "Avg IoU", "Class agreement", "Extra Veh generated", "Missing Veh generated", "Mean AP" "confidence difference"])
        for row in csv_rows:
            writer.writerow(row)

    global_csv_rows.append([ simulated_folder, np.mean(veh_offset), np.mean(recalls), np.mean(precisions), np.mean(avg_ious), np.mean(class_accuracies), np.mean(all_matched_gen_confidences), np.mean(confidence_differences), np.mean(mean_aps) ])

with open(csv_filename, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Version", "Veh offset mean", "Recalls", "Precisions", "Avg. IoU", "Class Accuracy Confidence", "Conf Diff Avg NoMean", "Conf Difference Avg", "Mean APS" ])
    for row in global_csv_rows:
        writer.writerow(row)



