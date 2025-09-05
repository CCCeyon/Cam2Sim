import os

import math
import osmnx as ox
import requests
from pyproj import Geod
from shapely.geometry import LineString, Point
import geopandas as gpd

from utils.debug import debug_loading_osm_data

geod = Geod(ellps="WGS84")

def get_origin_lat_lon(edges, address):
    """Convert a Point to (lat, lon) tuple."""
    center_point = edges.union_all().centroid
    origin_lat = center_point.y
    origin_lon = center_point.x
    return origin_lat, origin_lon

    #point = ox.geocoder.geocode(address)
    #print(point)
    #lat, lon = point
    #return lat, lon

    #return center_point.y, center_point.x

#def get_street_data2(address, dist=500):
#    debug_loading_osm_data(address)
#    ox.settings.all_oneway = True
#    G = ox.graph.graph_from_address(address, dist=dist, network_type="drive", simplify=False, retain_all=True)
#    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

#    buildings = ox.features_from_address(address, tags={"building": True}, dist=dist)

#    edges = edges.to_crs("EPSG:4326")
#    buildings = buildings.to_crs("EPSG:4326")

#    return G, edges, buildings

def get_street_data(address, dist=500):
    # 1. Geokodierung
    debug_loading_osm_data(address)
    point = ox.geocode(address)

    # 2. Bounding Box (rückgibt: north, south, east, west)
    west, south, east, north = ox.utils_geo.bbox_from_point(point, dist=dist)

    # 3. Overpass-Abfrage: beachte Reihenfolge (south, west, north, east)
    query = f"""
    [out:xml][timeout:25];
    (
      way["building"]({south},{west},{north},{east});
      way["highway"]({south},{west},{north},{east});
      relation["building"]({south},{west},{north},{east});
      relation["highway"]({south},{west},{north},{east});
    );
    (._;>;);
    out body;
    """

    # 4. Anfrage senden
    response = requests.get("https://overpass-api.de/api/interpreter", params={'data': query})

    # 5. Fehlerseite erkennen
    if not response.ok or not response.content.startswith(b"<?xml"):
        raise ValueError("⚠️ Overpass-Request failed. Error: ",response.content)

    return response.content  # bytes

def generate_spawn_gdf(edges, offset=4.0, override = False):
    lines = []
    sides = []
    for idx, row in edges.iterrows():
        geom = row.geometry

        if not isinstance(geom, LineString):
            continue

        # Parking tags (can be 'yes', 'parallel', 'diagonal', etc.)
        park_tags = {
            "parking:left": str(row.get("parking:left", "")).lower(),
            "parking:right": str(row.get("parking:right", "")).lower(),
            "parking:both": str(row.get("parking:both", "")).lower(),
            "parking:lane:left": str(row.get("parking:lane:left", "")).lower(),
            "parking:lane:right": str(row.get("parking:lane:right", "")).lower(),
            "parking:lane:both": str(row.get("parking:lane:both", "")).lower(),
        }

        # set park_accept to accept lane parking
        park_accept = {"yes", "parallel", "diagonal", "lane", "parallel:both", "diagonal:both", "lane:both", "on_street", "on_kerb"}

        left_allowed = (
                park_tags["parking:left"] in park_accept or
                park_tags["parking:lane:left"] in park_accept or
                park_tags["parking:both"] in park_accept or
                park_tags["parking:lane:both"] in park_accept or
                override
        )

        # Prüfe rechtes Parken erlaubt
        right_allowed = (
                park_tags["parking:right"] in park_accept or
                park_tags["parking:lane:right"] in park_accept or
                park_tags["parking:both"] in park_accept or
                park_tags["parking:lane:both"] in park_accept or
                override
        )

        # Create parallel offsets conditionally
        if left_allowed:
            left = geom.parallel_offset(offset / 111.32e3, 'left', join_style=2)
            if isinstance(left, LineString):
                lines.append(left)
                sides.append("left")

        if right_allowed:
            right = geom.parallel_offset(offset / 111.32e3, 'right', join_style=2)
            if isinstance(right, LineString):
                lines.append(right)
                sides.append("right")

    return gpd.GeoDataFrame({'geometry': lines, 'side': sides}, crs="EPSG:4326")

def get_heading(start_lat, start_lon, end_lat, end_lon):
    az, _, _ = geod.inv(start_lon, start_lat, end_lon, end_lat)
    return (az + 360) % 360

def latlon_to_carla(origin_lat, origin_lon, lat, lon):
    az12, az21, dist = geod.inv(origin_lon, origin_lat, lon, lat)
    rad = math.radians(az12)
    dx = math.cos(rad) * dist
    dy = math.sin(rad) * dist
    return (dx, dy, 0.0)

#def save_graph_to_osm(filepath, G):
#    ox.io.save_graph_xml(G, filepath=filepath)

def fetch_osm_data(map_folder):
    map_path = os.path.join(map_folder, "map.osm")
    G = ox.graph.graph_from_xml(map_path, simplify=False, retain_all=True)

    all_tags = [
        "parking:left", "parking:right", "parking:both",
        "parking:lane:left", "parking:lane:right", "parking:lane:both"
    ]
    tag_dict = {tag: True for tag in all_tags}
    parking_data = ox.features_from_xml(map_path, tags=tag_dict).reset_index()

    available_columns = [col for col in all_tags if col in parking_data.columns]
    parking_info = parking_data[["id"] + available_columns]
    parking_info["id"] = parking_info["id"].astype(str)

    _, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    tags = {"building": True}
    buildings = ox.features_from_xml(map_path, tags=tags)

    allowed_highways = {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "residential", "unclassified", "service"
    }

    edges = edges[edges["highway"].isin(allowed_highways)]

    edges = edges[edges["highway"].isin(allowed_highways)]

    edges["osmid_str"] = edges["osmid"].astype(str)
    edges = edges.merge(parking_info, left_on="osmid_str", right_on="id", how="left")

    print(edges.columns)

    return G, edges, buildings