# The original dataset for CNPR is not in COCO format. We need to convert it to COCO format to use the COCO evaluator.
"""
- CNR-EXT:
    - train:
        - _annotations.coco.json
        - images
    - valid:
        - _annotations.coco.json
        - images
    - test:
        - _annotations.coco.json
        - images
"""
import os

def _remove_old_files_and_create_paths(output_dir):
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print("Removed folder: ", output_dir)
    train_path = os.path.join(output_dir, "train")
    valid_path = os.path.join(output_dir, "valid")
    test_path = os.path.join(output_dir, "test")
    os.makedirs(train_path)
    os.makedirs(valid_path)
    os.makedirs(test_path)
    print("Created folders: {}, {}, and {}".format(train_path, valid_path, test_path))

def _add_common_parts_to_json_files(data, train_valid_test_names):
    json_files = {}
    for train_valid_test_name in train_valid_test_names:
        new_data = {}
        new_data["info"] = data["info"]
        new_data["licenses"] = data["licenses"]
        new_data["categories"] = data["categories"]
        new_data["images"] = []
        new_data["annotations"] = []
        json_files[train_valid_test_name] = new_data
    return json_files

def _parse_cameras_csv_files(camera_csv_files_path_pattern, num_cameras):
    cameras_csv_dict = {}
    for x in range(num_cameras):
        camera_id = x + 1
        cameras_csv_dict[camera_id] = {}
        camera_csv_file_path = camera_csv_files_path_pattern.format(camera_id)
        with open(camera_csv_file_path, "r") as f:
            import csv
            csv_reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i == 0: # Skip the header
                    continue
                # print(row)
                slotId = row[0]
                x = int(row[1])
                y = int(row[2])
                w = int(row[3])
                h = int(row[4])
                cameras_csv_dict[camera_id][slotId] = { "x": x, "y": y, "w": w, "h": h }
    return cameras_csv_dict

def transform_into_COCO(pklot_anno_path, train_percent, val_percent, test_percent, camera_csv_files_path_pattern, num_cameras, annotations_csv_file_path, output_dir):
    _remove_old_files_and_create_paths(output_dir)
    train_valid_test_names = ['train', 'valid', 'test']
    import random
    random.seed(17) # To get the same results from the random generator
    with open(pklot_anno_path, "r") as f:
        import json
        pklot_coco_data = json.load(f)

        json_files = _add_common_parts_to_json_files(pklot_coco_data, train_valid_test_names)
        cameras_csv_dict = _parse_cameras_csv_files(camera_csv_files_path_pattern, num_cameras)

        # Extract the directory path from `camera_csv_files_path_pattern`
        images_dir = os.path.join(os.path.dirname(camera_csv_files_path_pattern), "FULL_IMAGE_1000x750")
        # Loop through all the directories many levels deep until we reach the jpg images
        import glob
        images = glob.glob(os.path.join(images_dir, "**/*.jpg"), recursive=True)
        global_image_ids = {'train': 0, 'valid': 0, 'test': 0}
        img_name_to_id = {}
        img_name_to_set_selector = {} # To keep track if this image is used for train / valid / test
        for image_name in images:
            train_valid_test_selector = random.choices(train_valid_test_names, [train_percent, val_percent, test_percent])[0] # Random choice based on the weights
            if not (train_valid_test_selector == 'train'):
                x = 3
            image_dict = {
                "id": global_image_ids[train_valid_test_selector],
                "file_name": image_name,
                "width": 1000,
                "height": 750,
                "date_captured": "2024-06-28 19:00:00", # TODO: AB: Fix this
            }
            img_name_to_id[image_name] = global_image_ids[train_valid_test_selector]
            img_name_to_set_selector[image_name] = train_valid_test_selector
            json_files[train_valid_test_selector]["images"].append(image_dict)
            global_image_ids[train_valid_test_selector] += 1

    global_annotation_ids = {'train': 0, 'valid': 0, 'test': 0}
    with open(annotations_csv_file_path, "r") as f:
        import csv
        csv_reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            camera_id = row[0] # Not used
            if camera_id == 'A' or camera_id == 'B':
                continue # Not handled for now, as it is part of the original dataset # TODO: AB: Handle this
            image_url = row[4]
            is_occupied = int(row[7]) # 0 or 1
            slot_id = row[8]
            modified_image_url = image_url.replace("CNR-EXT/PATCHES/", "") # Remove the first character which is the weather letter, then underscore
            # Find the last slash index
            last_slash_index = modified_image_url.rfind('/')
            image_folder = modified_image_url[:last_slash_index]
            image_name = modified_image_url[last_slash_index + 3:] # To remove the first 2 characters of the name
            image_name = image_name[:16].replace(".", "")
            image_full_path = os.path.join(images_dir, image_folder, image_name + ".jpg")
            image_id = img_name_to_id[image_full_path]
            train_valid_test_selector = img_name_to_set_selector[image_full_path]
            lot_information = cameras_csv_dict[int(camera_id)][slot_id] # TODO: AB: This might be changed later to string
            area = lot_information["w"] * lot_information["h"]
            an_annotation = {
                "id": global_annotation_ids[train_valid_test_selector],
                "image_id": image_id,
                "category_id": is_occupied + 1,
                "bbox": [
                    lot_information["x"],
                    lot_information["y"],
                    lot_information["w"],
                    lot_information["h"]
                ],
                "area": area,
                "segmentation": [],
                "iscrowd": 0
            }
            json_files[train_valid_test_selector]["annotations"].append(an_annotation)
            global_annotation_ids[train_valid_test_selector] += 1
    
    # Write JSON files
    import json
    train_json_file_path = os.path.join(output_dir, "train/_annotations.coco.json")
    valid_json_file_path = os.path.join(output_dir, "valid/_annotations.coco.json")
    test_json_file_path = os.path.join(output_dir, "test/_annotations.coco.json")
    json_files_paths = [train_json_file_path, valid_json_file_path, test_json_file_path]
    for json_file, json_file_path in zip(json_files.values(), json_files_paths):
        with open(json_file_path, "w") as f:
            json.dump(json_file, f)
            print("Done writing to: ", json_file_path)

    from split_pklot_by_parking_lot import analyze_coco_file

    for json_file_path in json_files_paths:
        analyze_coco_file(json_file_path)

if __name__ == "__main__":
    test_annotations_pklot_path = '/home/bakr/pklot/test/_annotations.coco.json'
    output_dir = '/home/bakr/CNR-EXT'
    train_percent = 0.8
    val_percent = 0.1
    test_percent = 0.1
    camera_csv_files_path_pattern = '/home/bakr/CNR-EXT_FULL_IMAGE_1000x750/camera{}.csv'
    annotations_csv_file_path = '/home/bakr/CNR-EXT_FULL_IMAGE_1000x750/CNRPark+EXT.csv'
    num_cameras = 9

    transform_into_COCO(test_annotations_pklot_path, train_percent, val_percent, test_percent, camera_csv_files_path_pattern, num_cameras, annotations_csv_file_path, output_dir)
