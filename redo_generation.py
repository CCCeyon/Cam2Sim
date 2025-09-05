import os

#import numpy as np

from config import OUTPUT_FOLDER_NAME
from utils.argparser import parse_generation_args
#from utils.carla_simulator import carla_image_to_pil
#from utils.distortion import compute_intrinsic_matrix, simulate_distortion_from_pinhole
from utils.pygame_helper import setup_pygame, show_image, combine_images
from utils.save_data import create_output_folders, get_model_data, get_saved_arguments
from natsort import natsorted
from PIL import Image
from torchvision import transforms

from utils.stable_diffusion import generate_image, load_stable_diffusion_pipeline

args = parse_generation_args()

output_dir = os.path.join(OUTPUT_FOLDER_NAME, args.output_dir)
if not os.path.exists(output_dir):
    print(f"Output folder '{output_dir}' does not exist.")
    exit(1)

output_args = get_saved_arguments(output_dir)
if output_args is None:
    print(f"No saved arguments found in '{output_dir}'.")
    exit(1)

model_data = get_model_data(output_args["model"])
if model_data is None:
    print(f"Model '{output_args['model']}' not found.")
    exit(1)

pipe = load_stable_diffusion_pipeline(output_args["model"], model_data)

create_output_folders(output_dir, False)

rgb_output_folder = os.path.join(output_dir, "carla")
seg_output_folder = os.path.join(output_dir, "seg")
gen_output_folder = os.path.join(output_dir, "output")

pygame_screen, pygame_clock = setup_pygame(model_data["size"]["x"], model_data["size"]["y"])

split_value = args.split / 100
print("Split values: ", split_value if split_value < 1.0 else 0.99, split_value if split_value > 0.0 else 0.01)

prev_image = None
max_images = args.max_images
amount = 0
# for every image in seg_output_folder sorted by name using natsort
for filename in natsorted(os.listdir(seg_output_folder)):
    if filename.endswith(".png"):
        rgb_image_path = os.path.join(rgb_output_folder, filename)
        seg_image_path = os.path.join(seg_output_folder, filename)
        gen_image_path = os.path.join(gen_output_folder, filename)

        seg_image = Image.open(seg_image_path)

        image_transforms = transforms.Compose(
        [
            transforms.Resize(model_data["size"]["x"], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(model_data["size"]["x"]),
        ]
        )
        seg_image = image_transforms(seg_image)

        rgb_image = Image.open(rgb_image_path)
        rgb_image = image_transforms(rgb_image)
        #rgb_image_pil = carla_image_to_pil(rgb_image)


        generated_image = generate_image(pipe, seg_image, model_data, prev_image, split = args.split, set_seed = args.set_seed, guidance = args.guidance, rotate = args.rotate)
        prev_image = generated_image
        #prev_image = rgb_image
        combined = combine_images(rgb_image, generated_image)

        generated_image.save(gen_image_path)

        show_image(pygame_screen, combined)
        pygame_clock.tick(30)

        print(f"Processing {seg_image_path}")

        amount = amount + 1
        if max_images is not None and amount >= max_images:
            break



