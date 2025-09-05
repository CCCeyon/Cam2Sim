import carla
import time
import os
import numpy as np
import cv2
from PIL import Image

from dotenv import load_dotenv
from huggingface_hub import login
from config import HERO_VEHICLE_TYPE, CARLA_IP, CARLA_PORT, ROTATION_DEGREES, OUTPUT_FOLDER_NAME
from utils.argparser import parse_testing_args
from utils.carla_simulator import spawn_additional_vehicles, update_synchronous_mode, remap_segmentation_colors, \
    carla_image_to_pil, remove_sensor, delete_all_vehicles, get_spectator_transform, set_spectator_transform, \
    generate_world_map, get_rotation_matrix, get_translation_vector, spawn_parked_cars, get_transform_position
from utils.pygame_helper import combine_images, setup_pygame, setup_sensor, get_sensor_blueprint, \
    show_image
from utils.save_data import create_dotenv, get_map_data, create_output_folders, get_model_data, save_arguments
from utils.stable_diffusion import load_stable_diffusion_pipeline, generate_image
from utils.distortion import compute_intrinsic_matrix, simulate_distortion_from_pinhole
import torchvision.transforms as transforms

from utils.yolo import calculate_yolo_image, load_yolo_model

args = parse_testing_args()

if not args.only_carla:
    create_dotenv()
    load_dotenv()
    huggingface_token = os.getenv("HF_TOKEN")
    if not huggingface_token:
        raise ValueError("⚠️ Huggingface Token not found. Please set the HF_TOKEN in your .env file.")

    login(token=huggingface_token)

client = carla.Client(args.carla_ip,args.carla_port)
client.set_timeout(10.0)

tm = client.get_trafficmanager(8000)

world = client.get_world()
spectator_transform = get_spectator_transform(world)

model_data = get_model_data(args.model)

#model_data["camera"]["position"]["z"] = 1.262061
#model_data["camera"]["position"]["x"] = 0.82

# model_data["size"]["x"] = 1280
# model_data["size"]["y"] = 960

if model_data is None:
    print(f"Model '{args.model}' not found.")
    exit(1)

map_data = get_map_data(args.map, (model_data["size"]["x"], model_data["size"]["y"]) )

if not map_data:
    print(f"⚠️ Could not find complete map data for {args.map}. Please check the map name or download the map data.")
    exit(1)

offset_data = map_data["vehicle_data"]["offset"]
if offset_data["x"] is None or offset_data["y"] is None or offset_data["x"] is None:
    print(f"⚠️ Offset Settings for {args.map} are not set. Please check the vehicle_data.json and set x,y and heading values to align vehicles with the carla map")
    exit(1)

world = generate_world_map(client, map_data["xodr_data"])

set_spectator_transform(world, spectator_transform)
blueprint_library = world.get_blueprint_library()

rotation_matrix = get_rotation_matrix(ROTATION_DEGREES)
translation_vector = get_translation_vector(map_data["xodr_data"], offset_data)

starting_location = get_transform_position(map_data["vehicle_data"]["hero_car"]["position"], translation_vector, rotation_matrix)

starting_position = carla.Transform(
    carla.Location(x = starting_location[0], y = starting_location[1], z = starting_location[2]),
    carla.Rotation(pitch=0, yaw=(map_data["vehicle_data"]["hero_car"]["heading"] + ROTATION_DEGREES) % 360, roll=0)
)

current_pos = spectator_transform.location
print("CURRENT POS: ",current_pos)
if abs(current_pos.x - (-172.196)) < 0.5 and abs(current_pos.y - (183.859)) < 0.5:
    print("⚠️ Spectator is very close to default position. Adjusting position.")
    set_spectator_transform(world, starting_position)

update_synchronous_mode(world, tm, True, model_data["camera"]["fps"])


vehicle_bp = blueprint_library.find(HERO_VEHICLE_TYPE)
vehicle = world.spawn_actor(vehicle_bp, starting_position)

blacklisted = [ "vehicle.carlamotors.carlacola", "vehicle.micro.microlino", "vehicle.harley-davidson.low_rider", "vehicle.kawasaki.ninja", "vehicle.vespa.zx125", "vehicle.yamaha.yzf", "vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets" ]
vehicle_library = [bp for bp in blueprint_library.filter('*vehicle*') if (bp.has_attribute('generation') and bp.get_attribute('generation').as_int() == 1 and bp.id not in blacklisted) or bp.id == "vehicle.mercedes.sprinter" ]  
#vehicle_library = blueprint_library.filter("vehicle.mini.cooper_s")
# remove blacklisted and all motorcycles from vehicle_library

spawn_parked_cars(world, vehicle_library, map_data["vehicle_data"]["spawn_positions"], translation_vector, rotation_matrix)
spawn_additional_vehicles(world, vehicle_library, args.other_vehicles)

print(model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"])
seg_sensor_bp = get_sensor_blueprint(blueprint_library,'sensor.camera.semantic_segmentation',model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])
rgb_sensor_bp = get_sensor_blueprint(blueprint_library,'sensor.camera.rgb',model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])
depth_sensor_bp = get_sensor_blueprint(blueprint_library,'sensor.camera.depth',model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])

sensor_spawn_point = carla.Transform(
    carla.Location(
        x=model_data["camera"]["position"]["x"],
        y = model_data["camera"]["position"]["y"],
        z=model_data["camera"]["position"]["z"]
    ), 
    carla.Rotation(pitch=0.0)
)

#carla.Rotation(pitch=model_data["camera"]["pitch"])
seg_sensor, seg_queue = setup_sensor(seg_sensor_bp, sensor_spawn_point, vehicle, world)
rgb_sensor, rgb_queue = setup_sensor(rgb_sensor_bp, sensor_spawn_point, vehicle, world)
depth_sensor, depth_queue = setup_sensor(depth_sensor_bp, sensor_spawn_point, vehicle, world)

vehicle.set_autopilot(True)

pygame_screen, pygame_clock = setup_pygame(model_data["size"]["x"], model_data["size"]["y"], args.only_carla)

if not args.no_save:
    output_dir = args.output_dir
    if not output_dir:
        output_dir = "output_" + str(int(time.time()))
    output_dir = os.path.join(OUTPUT_FOLDER_NAME,output_dir)
    create_output_folders(output_dir)
    save_arguments(args, output_dir)

if not args.only_carla:
    pipe = load_stable_diffusion_pipeline(args.model, model_data)
    prev_image = map_data["starting_image"]
    yolo_model = load_yolo_model()


image_transforms = transforms.Compose(
        [
            transforms.Resize(model_data["size"]["x"], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(model_data["size"]["x"]),
        ]
        )

frame = world.tick()
kill_frame = frame + (args.seconds * model_data["camera"]["fps"])
try:
     while True:
        frame = world.tick()
        #for _ in range(10):
        #    frame = world.tick()

        seg_image = seg_queue.get()
        rgb_image = rgb_queue.get()
        depth_image = depth_queue.get()

        rgb_image_pil = carla_image_to_pil(rgb_image)

        seg_image.convert(carla.ColorConverter.CityScapesPalette)
        seg_image = carla_image_to_pil(seg_image)
        seg_image = remap_segmentation_colors(seg_image)

        if "calibration" in model_data["camera"].keys() :
            print("Running distortion")
            K_pinhole = compute_intrinsic_matrix(model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])    # Replace with your real camera intrinsics (e.g., from checkerboard calibration)
            K_real = np.array(model_data["camera"]["calibration"]["K"], dtype=np.float32)    # Real distortion coefficients [k1, k2, p1, p2, k3]
            dist_real = np.array(model_data["camera"]["calibration"]["distortion"], dtype=np.float32)    # Simulate real camera distortion on CARLA image
            distorted_image = simulate_distortion_from_pinhole(seg_image, K_pinhole, K_real, dist_real)    # Show and save the result
            rgb_image_pil = simulate_distortion_from_pinhole(rgb_image_pil, K_pinhole, K_real, dist_real)    # Show and save the result
            rgb_image_pil.save("temp.png")

            distorted_image = image_transforms(distorted_image)
            rgb_image_pil = image_transforms(rgb_image_pil)

        else:
            distorted_image = image_transforms(seg_image)
        #distorted_image.show()
        #distorted_image.save("carla_image_distorted.png")


        depth_image.convert(carla.ColorConverter.LogarithmicDepth)
        depth_image = carla_image_to_pil(depth_image)
        #depth_image.show()

        depth_image = np.array(depth_image)
        kernel = np.ones((3,3),np.float32)/10
        depth_image = cv2.filter2D(depth_image,-1,kernel)
        depth_image = Image.fromarray(depth_image)



        #canny_image = get_canny_from_rgb(rgb_image,100,200)
        rgb_image_np = np.array(rgb_image_pil)
        canny_image = cv2.Canny(rgb_image_np, 100, 200)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        canny_image = Image.fromarray(canny_image)
        if not args.only_carla:
            generated_image = generate_image(pipe, distorted_image, model_data, prev_image, split = 30, guidance = 3.5, set_seed = True, rotate = False)
            prev_image = generated_image
            #generated_image.save("zzzzzz.png")
            _, yolo_image = calculate_yolo_image(yolo_model, generated_image)
        #generated_image = seg_image
        #args.no_save = False
            combined = combine_images(rgb_image_pil, yolo_image)
        else:
            combined = rgb_image_pil

        if not args.no_save:
            distorted_image.save(os.path.join(output_dir, "seg", '%06d.png' % frame))
            if not args.only_carla:
                generated_image.save(os.path.join(output_dir, "output", '%06d.png' % frame))
            rgb_image_pil.save(os.path.join(output_dir, "carla", '%06d.png' % frame))
            depth_image.save(os.path.join(output_dir, "depth", '%06d.png' % frame))
            depth_image.save(os.path.join(output_dir, "canny", '%06d.png' % frame))

        show_image(pygame_screen, combined)
        pygame_clock.tick(30)

        if frame >= kill_frame:
            print("Terminating simulation")
            break
            
        
finally:
    remove_sensor(seg_sensor)
    remove_sensor(depth_sensor)
    remove_sensor(rgb_sensor)
    update_synchronous_mode(world, tm, False)
    delete_all_vehicles(world)