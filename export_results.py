import cv2
import os
from natsort import natsorted  # Für natürliche Sortierung (z. B. 1, 2, 10 statt 1, 10, 2)

from config import VIDEO_EXPORT_FOLDER
from utils.argparser import parse_export_args
from utils.export import images_to_video

args = parse_export_args()

output_folder_name = args.name #"output_1753369738"

os.makedirs(VIDEO_EXPORT_FOLDER, exist_ok=True)
export_path = os.path.join(VIDEO_EXPORT_FOLDER,f"{output_folder_name}.mp4")
images_to_video(output_folder_name, export_path, fps=24, dir = args.dir, only_generated = args.only_generated)
