# -------------------
# GENERAL SETTINGS
# -------------------
DEBUG_PRINT = True
CARLA_INSTALLATION_PATH = "/media/constantin/ExtraSSDSpace/Testing/Carla"
#CARLA_INSTALLATION_PATH = "/home/csys/Desktop/CARLA_0.9.15"
CARLA_GLASS_PATH = "CarlaUE4/Content/Carla/Static/Car/GeneralMaterials/Glass.uasset"
ASSET_PATH = "assets"
# -------------------
# MAP CREATION CONFIGURATION
# -------------------

HERO_CAR_OFFSET_METERS = 2.0 #2.0
SPAWN_OFFSET_METERS = 5.0 #5.5
CLICK_DISTANCE_THRESHOLD = 0.0001  # ~11m
MAPS_FOLDER_NAME = "maps"

# -------------------
# DATASET CREATION CONFIGURATION
# -------------------

CANNY_LOW_THRESHOLD = 120
CANNY_HIGH_THRESHOLD = 180
LLAVA_MODEL = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
SEGFORMER_MODEL = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
LLAVA_BATCH_SIZE = 20
CARLA_MODE = True
LLAVA_DESCRIPTION_PROMPT = "Tag the image seperated by commas, starting with street, and sort them by the amount of space they take. example tags: building, car, sidewalk, person, trees, etc. (if possible)"
DATASETS_FOLDER_NAME = "datasets"

# -------------------
# FINETUNING CONFIGURATION
# -------------------
MODEL_FOLDER_NAME = "models"
STABLE_DIFF_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMAGE_RESIZE = (512, 512)  # Resize images to 512x512 for Stable Diffusion

# -------------------
# TESTING CONFIGURATION
# -------------------

CARLA_IP = "localhost"
CARLA_PORT = 2000

HERO_VEHICLE_TYPE = 'vehicle.tesla.model3'
VERTEX_DISTANCE = 0.1  # in meters
MAX_ROAD_LENGTH = 500.0 # in meters
WALL_HEIGHT = 0.0      # in meters
EXTRA_WIDTH = 0.6      # in meters

FORWARDS_PARKING_PROBABILITY = 100 - 76 # in percent

CAR_SPACING = 5

ROTATION_DEGREES = 261
STABLE_DIFF_PROMPT = "street, car, sidewalk, trees, person, trees, cars, trees, cars, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees"
STABLE_DIFF_STEPS = 50
OUTPUT_FOLDER_NAME = "output"
SEGMENTATION_COND_SCALE = 0.8

# -------------------
# EXPORT CONFIGURATION
# -------------------

VIDEO_EXPORT_FOLDER = "exports"

# CHECKERBOARD

CHECKERBOARD_COLS = 10
CHECKERBOARD_ROWS = 7
CHECKERBOARD_MIN_FRAMES = 15