# CarlaVisTool

## Setup

Ensure you are working in a Python 3.9 environment. You can create a virtual environment using:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

### Carla Package (only required for Step 5)
The initial steps to create a Carla-Ready map, a dataset from a dashcam video and finetune a model do not
require carla to be installed, but the last step does require a running Carla server and the carla pip-package<br>
To install the Carla package, you can use the following command:
```bash
pip install carla
```
This will most likely not work on all machines, as the carla package is not available for all Python versions and architectures.<br>

## 1. Map Creation from an Address

To create a Carla-Ready map, you need to provide the Address (`address`) and distance (`dist`) 
of the map you want to create. The script will then generate a map in the `maps` directory.<br>

To help you fill the map with cars, the script will open a Topdown-View of the street, where you can select
the spawn-point of the hero-car and areas for parked cars, which will be filled with random cars.<br>
The first click will set the spawn-point of the hero-car, after that you can start an area with a click 
and end it with another click. Areas have a maximum size, as a area can only be filled from one node to the next.<br>

If you want to use a different name for the map, you can use the `--name` flag to specify the name of the map. 
Otherwise, the name will be generated from the address and date.<br>

If you dont have carla installed, you can use the `--no-carla` flag to skip the carla-related parts of the script.
This will not create the required `.xodr` file, but will save it as an `.osm` file instead, 
which will instead be translated to a `.xodr` file later on. (on a service that runs the final carla-map)<br>

The script can be run with the following command:
```bash
python create_map.py --address "Address" --dist 200 # --name "MapName" --no-carla
```

#### Example:
```bash
python create_map.py \
  --address "Guerickestraße, Alte Heide, Munich" \
  --dist 200 
```

## 2. Dataset Creation from a Dashcam Video
To create a dataset, you only need to provide the Video File (`video_file`) of the dashcam footage you want to use.
The script will separate the video into frames, create a segmentation mask for each frame 
and tag the frames using LlaVa (by default).<br>
If you wish, you can also have canny edges generated for each frame, which can be used to 
train a model that is able to detect edges in the video. This can be done by using the `--canny` flag.<br>

If you wish to upload the dataset to huggingface, you can use the `--upload` flag. 
The Huggingface token will need the `write:datasets` scope, otherwise the upload will fail.<br>

By default, the script will create a dataset in the `dataset` directory, 
with the name of the video file and the current date as the dataset name.<br>
If you want to use a different name, you can use the `--name` flag to specify the name of the dataset.
We recommend using the street name as the name of the dataset, as this will help you to find the dataset later on.<br>

As the frame extraction and segmentation can take a while, if the script finds a dataset with the same 
name in the `dataset` directory with the same amount of frames, it skip the frame extraction, canny edges,
and segmentation steps and only tag the frames using the already existing images.<br>

If you wish to blur faces in the video, you can use the `--blur` flag. 
Please note that this does not work reliably on all faces, as it uses a simple face detection algorithm.
If you wish to publish your dataset, make sure to check the dataset for any faces that were not blurred.<br>

It is strongly recommended to set the field of view of the camera in degrees using the `--fov` flag,
`--pitch` for the pitch of the camera in degrees 
aswell as the cameras position using `--pos_x`, `--pos_y` and `--pos_z`.
These attributes will be forwarded to the model later on, so it can generate images with the same field of view
and camera position as the video.<br>
You can change the values anytime later on in the models `config.json` file.<br>
If unset, the values will be defaulted to a Teslas Model 3 Dashcam.

The script can be run with the following command:
```bash
python create_dataset.py --video "Video.mp4" # --name "MyDataset" --upload --canny --blur --fov 90 --pos_x 0.0 --pos_y 0.0 --pos_z 0.0 --pitch 0.0
```

#### Example:
```bash
python create_dataset.py \
  --video "guerickestrasse.mp4" \
  --upload \
  --fov 55 \
  --pos_x 1.1 \
  --pos_z 1.35
```

## 3. Finetuning Models
To finetune a model, you need to provide the Dataset Name (`dataset`) of the dataset you want to use.
The script will then finetune a Stable Diffusion Model with lora using the dataset. (images and tags)<br>
Additionally, a Segmentation Model will be trained using the segmentation masks of the dataset.<br>

Both models will be saved in the `models` directory, with the name of the dataset as the model name<br>

The script can be run with the following command:
```bash
python finetune_model.py --dataset "DatasetName"
```

#### Example:
```bash
python finetune_model.py --dataset "guerickestrasse_25_07_01"
```

## 4. Starting the Carla Server

To start the Carla server, you need to have Carla installed and running on your machine.
You can set the Carla path in the `config.py` file or provide it as an argument when running the script.<br>

The script will replace the assets within Carla with the assets from the `assets/Carla` directory and revert
all changes after termination of the script.<br>

It is required to start Carla using this script OR manually replace the Assets in `assets/Carla`

You can start the server with the following command:
```bash
python setup_carla.py #--carla_path "PathToCarla"
```

## 5. Using the Model on the Map
To test the model on the map, you need to provide the Map Name (`map`) of the map 
and the Model Name (`model`) of the model you want to use. 
Both need to be in the `maps` and `models` directories, respectively.<br>

The script will then load the map and the model, spawn all cars in the map and 
use the model to generate images of the map. It will showcase the current generated image 
next to the segmentation mask in a window<br>

By default, it saves the generated images in the `output` directory, 
with the timestamp as its name. This can be disabled using the `--no_save` flag or changed using the `output_dir` argument.<br>

If you only want to generate the carla and segmentation images, you can use the `--only_carla` flag to skip using the model
and want to generate them afterwards using the `redo_generation` Script. This should only be used if your system is not
powerful enough to run carla and the Stable Diffusion Pipeline simultaneously.<br>

If done correctly, the output should resemble the video provided in the dataset creation section.<br>

Have a carla server running
and then run the script with the following command:
```bash
python test_model.py --map "MapName" --model "ModelName" # --other_vehicles 5 --no-save # --output_dir "FolderName" --only_carla
```

#### Example:
```bash
python test_model.py \
  --map "gurickestrae_alte_heide_munchen_25_07_01" \
  --model "guerickestrasse_25_07_01" \
  --other_vehicles 25 \
  --output_dir "FirstTestRun" 
```


### 5.1 Redo the Generation
If you want to overwrite the generated images or havent generated the images yet (using the previous `--only_carla` flag,
you can use the `redo_generation.py` script. You need to provide the Output Folder (`--output_dir`) where the images are stored.<br>
The script will, using the same model, generate the images (again) and save them in the same folder.<br>

If you just wish to generate new images without overwriting the old ones, you just need to run the previous script again
instead of this one. This is only intended for separating the load on the GPU if your hardware is limited.<br>
This script does not require a running carla server, nor the carla pip-package<br>

```bash
python redo_generation.py --output_dir "OutputFolderName"
```

#### Example:
```bash
python redo_generation.py \
  --output_dir "FirstTestRun" 
```