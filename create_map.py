import json
import os.path

from config import MAPS_FOLDER_NAME
from utils.argparser import parse_map_args
from utils.map_data import get_street_data, fetch_osm_data
from utils.other import ensure_carla_functionality
from utils.plotting import create_plot, show_plot, get_output
from utils.save_data import get_map_folder_name, create_map_folders, save_vehicle_data, save_map_data, save_osm_data, get_map_data, get_existing_osm_data

# -------------------
# PLOTTING MAP
# -------------------

args = parse_map_args()
map_name = args.name if args.name else get_map_folder_name(args.address)
folder_name = os.path.join(MAPS_FOLDER_NAME, map_name )
map_data = get_map_data(map_name, None, args.no_carla)

if not args.no_carla:
    ensure_carla_functionality()  # Ensure that Carla functionality is available if not disabled

#G, edges, buildings = get_street_data(args.address, dist=args.dist) # Load street data from OSM using the provided address and distance

if not args.skip_fetch:
    osm_data = get_street_data(args.address, dist=args.dist) # Load street data from OSM using the provided address and distance
else:
    osm_data = get_existing_osm_data(folder_name)
create_map_folders(folder_name)
if not args.skip_fetch:
    save_osm_data(folder_name, osm_data)
G, edges, buildings = fetch_osm_data(folder_name)
create_plot(buildings, edges, args.address) # Create the plot with buildings and edges
show_plot() # Show the plot and wait for user interaction

# -------------------
# SAVE OUTPUT
# -------------------

output_json = get_output(args.dist) # Get the output data after user interaction
if map_data is not None and map_data["vehicle_data"]["dist"] == args.dist and map_data["vehicle_data"] is not None and map_data["vehicle_data"]["offset"] is not None:
    output_json["offset"] = map_data["vehicle_data"]["offset"]
    print("\n⚠️ Copied Offset Values from existing Map-Data.")
    

if output_json is not None:
    save_map_data(folder_name, osm_data, args.no_carla)
    save_vehicle_data(folder_name, output_json)
    print("\n📂 Map saved successfully to:", folder_name)
else:
    print("\n⚠️ No Data available to save.")
