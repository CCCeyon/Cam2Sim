import os
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import Dataset

from config import DATASETS_FOLDER_NAME
from utils.argparser import parse_dataset_args
from utils.dataset import get_video_dataset, describe_images, get_dataset_features, create_segmentation_data, create_depth_data, create_canny_data
from utils.save_data import create_dataset_folders, get_dataset_folder_name, save_dataset, create_dotenv, \
    delete_image_files

args = parse_dataset_args()

create_dotenv()
load_dotenv()
huggingface_token = os.getenv("HF_TOKEN")
if not huggingface_token:
    raise ValueError("⚠️ Huggingface Token not found. Please set the HF_TOKEN in your .env file.")

login(token=huggingface_token)

dataset_name = args.name if args.name else get_dataset_folder_name(args.video)
dataset_path = os.path.join(DATASETS_FOLDER_NAME, dataset_name)

create_dataset_folders(dataset_path)
dataset = get_video_dataset(dataset_path, args.video, args.canny, args.blur, override = args.override)
create_segmentation_data(dataset,dataset_path)
if args.canny:
    create_canny_data(dataset,dataset_path)
create_depth_data(dataset,dataset_path)
describe_images(dataset)

dataset_features = get_dataset_features(args.canny)
hf_dataset = Dataset.from_list(dataset, features=dataset_features)

save_dataset(dataset_path, hf_dataset)
#delete_image_files(dataset_path)
print("\n📂 Dataset saved successfully to:", dataset_path)

if args.upload:
    hf_dataset.push_to_hub(dataset_name, private=True, token = huggingface_token)
    print("\n💻 Dataset privately uploaded to Huggingface as:", dataset_name)