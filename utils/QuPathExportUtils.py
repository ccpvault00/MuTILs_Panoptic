import os
import json
import numpy as np
import cv2
from PIL import Image
from shapely.geometry import Polygon
from shapely import affinity
from MuTILs_Panoptic.configs.panoptic_model_configs import RegionCellCombination

def export_binary_mask_to_geojson(binary_array, save_dir, roi_name, label_name,
                                     offset_x=0, offset_y=0, scale=1.0,
                                     min_area=10, debug=False):

    contours, hierarchy = cv2.findContours(binary_array, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return

    hierarchy = hierarchy[0]
    features = []
    skipped = 0
    total = 0

    for i, contour in enumerate(contours):
        total += 1
        area = cv2.contourArea(contour)
        if area < min_area:
            skipped += 1
            continue

        contour = contour.squeeze(1)
        if contour.ndim != 2 or contour.shape[0] < 3:
            skipped += 1
            continue
        if hierarchy[i][3] != -1:
            continue

        exterior = contour
        holes = [
            contours[j].squeeze(1)
            for j in range(len(contours))
            if hierarchy[j][3] == i and len(contours[j]) >= 3
        ]

        exterior_coords = [[float(x)*scale + offset_x, float(y)*scale + offset_y] for x, y in exterior]
        holes_coords = [
            [[float(x)*scale + offset_x, float(y)*scale + offset_y] for x, y in hole]
            for hole in holes if len(hole) >= 3
        ]

        try:
            poly = Polygon(exterior_coords, holes_coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or not poly.is_valid:
                skipped += 1
                continue
            features.append({
                "type": "Feature",
                "geometry": poly.__geo_interface__,
                "properties": {
                    "classification": {
                        "name": label_name,
                        "colorRGB": -1
                    }
                }
            })
        except Exception as e:
            skipped += 1
            if debug:
                print(f"[⚠️] Polygon skipped: {e}")
            continue

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{roi_name}_{label_name}_.geojson"), 'w') as f:
        json.dump(geojson, f, indent=2)

    if debug:
        print(f"[✅] {label_name}: Total contours: {total}, Exported: {len(features)}, Skipped: {skipped}")

def export_region_masks_to_geojson(region_mask, region_code_map, save_dir,
                                  roi_name, offset_x=0, offset_y=0, scale=1.0,
                                  min_area=10, debug=False):
    for label_name, label_id in region_code_map.items():
        if label_id in [0, 9]:
            continue
        binary = (region_mask == label_id).astype(np.uint8)
        export_binary_mask_to_geojson(
            binary_array=binary,
            save_dir=save_dir,
            roi_name=roi_name,
            label_name=label_name,
            offset_x=offset_x,
            offset_y=offset_y,
            scale=scale,
            min_area=min_area,
            debug=debug
        )
