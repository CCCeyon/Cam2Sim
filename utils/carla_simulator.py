import random
import carla
import xml.etree.ElementTree as ET

import math
import numpy as np
from PIL import Image

from config import VERTEX_DISTANCE, MAX_ROAD_LENGTH, WALL_HEIGHT, EXTRA_WIDTH, ROTATION_DEGREES, CAR_SPACING, \
    FORWARDS_PARKING_PROBABILITY


def get_map_size(xodr_data):
    root = ET.fromstring(xodr_data)
    header = root.find("header")

    north = float(header.attrib.get("north", "0"))
    south = float(header.attrib.get("south", "0"))
    east = float(header.attrib.get("east", "0"))
    west = float(header.attrib.get("west", "0"))

    width = east - west
    height = north - south

    return width, height

def spawn_additional_vehicles(world, vehicle_blueprints, vehicle_amount):
    if vehicle_amount <= 0:
        return
    spawn_points = world.get_map().get_spawn_points()
    for i in range(0,vehicle_amount):
        world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

    for vehicle in world.get_actors().filter('*vehicle*'):
        vehicle.set_autopilot(True)

def update_synchronous_mode(world, tm, bool, fps = 1):
    settings = world.get_settings()
    if bool:
        settings.fixed_delta_seconds = 1 / fps
    settings.synchronous_mode = bool
    world.apply_settings(settings)

    tm.set_synchronous_mode(bool)

def remap_segmentation_colors(seg_image):
    arr = np.array(seg_image)
    # 0,0,0 → 128,64,128
    mask1 = np.all(arr == [0, 0, 0], axis=-1)
    arr[mask1] = [128, 64, 128]
    # 70,130,180 → 0,0,0
    mask2 = np.all(arr == [70, 130, 180], axis=-1)
    arr[mask2] = [0, 0, 0]
    return Image.fromarray(arr)

def carla_image_to_pil(image) -> Image.Image:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    rgb_array = array[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB
    return Image.fromarray(rgb_array)

def remove_sensor(sensor):
    if sensor is not None:
        sensor.stop()
        sensor.destroy()

def delete_all_vehicles(world):
    actors = world.get_actors().filter('vehicle.*')
    for actor in actors:
        actor.destroy()

def get_spectator_transform(world):
    spectator = world.get_spectator()
    transform = spectator.get_transform()
    return transform

def set_spectator_transform(world, transform):
    spectator = world.get_spectator()
    spectator.set_transform(transform)

def generate_world_map(client, xodr_data):
    return client.generate_opendrive_world(
        xodr_data,
        carla.OpendriveGenerationParameters(
            vertex_distance=VERTEX_DISTANCE,
            max_road_length=MAX_ROAD_LENGTH,
            wall_height=WALL_HEIGHT,
            additional_width=EXTRA_WIDTH,
            smooth_junctions=True,
            enable_mesh_visibility=True
        )
    )

def get_rotation_matrix(roation_degrees):
    theta = math.radians(roation_degrees)
    return np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])

def get_translation_vector(xodr_data, offset):
    """ Gibt den Translationsvektor für die Verschiebung in X, Y zurück. """
    size_x, size_y = get_map_size(xodr_data)

    center = np.array([offset["x"], offset["y"]])
    rotation_matrix = get_rotation_matrix(offset["heading"])
    rotated_center = rotation_matrix @ center

    translation = -rotated_center

    #print("ORIGINAL:", center)
    #print("ROTATED: ", rotated_center)
    #print("TRANSLATION VECTOR: ",translation)

    #return np.array(rotated_center[0], -rotated_center[1])
    #return np.array([292.5920715332031, -226.9042205810547])  # Verschiebung in X, Y (nach der Rotation)
    #return np.array([size_x / 2 + 5.42, size_y / -2 + 50.57])  # Verschiebung in X, Y (nach der Rotation)
    #return np.array([size_x / 2, size_y / -2])  # Verschiebung in X, Y (nach der Rotation)
    #print(np.array([size_x / 2 + translation[0], size_y / -2 + translation[1]]))
    return np.array([size_x / 2 + translation[0], size_y / -2 + translation[1]])  # Verschiebung in X, Y (nach der Rotation)

def get_transform_position(pos, translation_vector, rotation_matrix):
    """ wendet Rotation + Translation auf eine [x, y, z] Position an """
    xy = np.array(pos[:2])
    rotated = rotation_matrix.dot(xy)
    translated = rotated + translation_vector
    return [translated[0], translated[1], pos[2]]

def spawn_parked_cars(world, vehicle_library, spawn_positions, translation_vector, rotation_matrix):
    for entry in spawn_positions:

        start = get_transform_position(entry["start"], translation_vector, rotation_matrix)
        end = get_transform_position(entry["end"], translation_vector, rotation_matrix)
        heading = (entry["heading"] + ROTATION_DEGREES) % 360

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx ** 2 + dy ** 2)

        if length < 0.1:
            print("⚠️ Skipping very short segment.")
            continue

        ux, uy = dx / length, dy / length
        num_cars = int(length // CAR_SPACING)

        if num_cars == 0:
            print("⚠️ Too short for parked car.")
            continue

        for i in range(num_cars):
            blueprint = random.choice(vehicle_library)
            #blueprint = blueprint_library.find('vehicle.tesla.model3')

            x = start[0] + i * CAR_SPACING * ux
            y = start[1] + i * CAR_SPACING * uy
            z = start[2] if len(start) > 2 else 0.0

            veh_heading = heading
            if entry["mode"] == "perpendicular" and random.random() < ( FORWARDS_PARKING_PROBABILITY / 100 ):
                veh_heading = (veh_heading + 180) % 360

            location = carla.Location(x=x, y=y, z=z)
            rotation = carla.Rotation(yaw=veh_heading)
            transform = carla.Transform(location, rotation)

            try:
                actor = world.spawn_actor(blueprint, transform)
                actor.set_simulate_physics(False)
                actor.set_target_velocity(carla.Vector3D(0, 0, 0))
                actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                #print(f"✅ Parked car at ({x:.1f}, {y:.1f}) heading {heading:.1f}°")
            except RuntimeError as e:
                print(f"❌ Could not spawn car at ({x:.1f}, {y:.1f}): {e}")
