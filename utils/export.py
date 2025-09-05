import os
import cv2
from natsort import natsorted
import numpy as np
import torchvision.transforms as transforms
from config import OUTPUT_FOLDER_NAME
from PIL import Image

def resize_and_center_crop(img, size=512):
    # Resize (keep aspect ratio like torchvision.Resize does)
    h, w = img.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop
    h, w = img_resized.shape[:2]
    top = (h - size) // 2
    left = (w - size) // 2
    img_cropped = img_resized[top: top + size, left: left + size]

    return img_cropped

def images_to_video(image_folder, output_path, fps=24, dir = "output", only_generated = False):
    carla_folder = os.path.join(OUTPUT_FOLDER_NAME, image_folder, "carla")
    output_folder = os.path.join(OUTPUT_FOLDER_NAME, image_folder, dir)

    # Hole Dateinamen aus beiden Ordnern
    if not only_generated:
        carla_images = [f for f in os.listdir(carla_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        carla_images = natsorted(carla_images)

    output_images = [f for f in os.listdir(output_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    output_images = natsorted(output_images)

    # Für Matching: nur die Schnittmenge der Dateinamen verwenden
    if not only_generated:
        common_images = sorted(set(carla_images) & set(output_images))
        if not common_images:
            print("Keine gemeinsamen Bilder gefunden.")
            return False
    else:
        common_images = sorted(output_images)

    # Erste Bildpaare laden, um Größe zu bestimmen
    carla_first = cv2.imread(os.path.join(carla_folder, common_images[0]))
    output_first = cv2.imread(os.path.join(output_folder, common_images[0]))

    if carla_first is None or output_first is None:
        print("Fehler beim Laden der ersten Bilder.")
        return False

    # Höhe anpassen, falls unterschiedlich (einfachheitshalber Höhe angleichen, Breite bleibt original)
    h = min(carla_first.shape[0], output_first.shape[0])
    carla_first = cv2.resize(carla_first, (int(carla_first.shape[1] * h / carla_first.shape[0]), h))
    output_first = cv2.resize(output_first, (int(output_first.shape[1] * h / output_first.shape[0]), h))

    if not only_generated:
        width = carla_first.shape[1] + output_first.shape[1]
    else:
        width = 1024 #output_first.shape[1]
        h = 1024
    height = h
    size = (width, height)

    # VideoWriter initialisieren
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    image_transforms = transforms.Compose([
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(1024),
    ])


    print(f"[*] Exportiere {len(common_images)} Bildpaare mit {fps} FPS zu {output_path}")
    for img_name in common_images:
        if not only_generated:
            output_img = cv2.imread(os.path.join(output_folder, img_name))
            carla_img = cv2.imread(os.path.join(carla_folder, img_name))

            if carla_img is None or output_img is None:
                print(f"[!] Bild {img_name} fehlt in einem Ordner, übersprungen.")
                continue

            # Höhe angleichen
            h = min(carla_img.shape[0], output_img.shape[0])
            carla_img = cv2.resize(carla_img, (int(carla_img.shape[1] * h / carla_img.shape[0]), h))
            output_img = cv2.resize(output_img, (int(output_img.shape[1] * h / output_img.shape[0]), h))

            combined = np.hstack((carla_img, output_img))

            out.write(combined)
        else:
            pil_img = Image.open(os.path.join(output_folder, img_name)).convert("RGB")

            # Apply transform
            pil_img = image_transforms(pil_img)
            output_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            #output_img = resize_and_center_crop(output_img, 512)
            out.write(output_img)

    out.release()
    return True
