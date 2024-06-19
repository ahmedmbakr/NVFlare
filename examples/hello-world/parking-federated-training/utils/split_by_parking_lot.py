"""
The initial structure of the PkLot dataset is as follows: train, valid, and test folder. Each one has "_annotations.coco.json" file and all the images.
What this script does is to split the dataset into parking lots. The dataset is split into 3 parking lots: PUCPR, UFPR04, and UFPR05.
The new structure will be as follows:
- PUCPR
    - train
        - _annotations.coco.json
        - images
    - valid
        - _annotations.coco.json
        - images
    - test
        - _annotations.coco.json
        - images
- UFPR04
    - train
        - _annotations.coco.json
        - images
    - valid
        - _annotations.coco.json
        - images
    - test
        - _annotations.coco.json
        - images
- UFPR05
    - train
        - _annotations.coco.json
        - images
    - valid
        - _annotations.coco.json
        - images
    - test
        - _annotations.coco.json
        - images
"""
import sys, os
import json
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '../')))

import pklot_trainer_config as config

parking_lot_to_dates_dict = { "PUCPR": ("2012-09-11", "2012-11-20"),
                                "UFPR04": ("2012-12-07", "2013-01-29"),
                                "UFPR05": ("2013-02-22", "2013-04-16")
                            }

parking_lot_to_json_dict = {
    "PUCPR": {},
    "UFPR04": {},
    "UFPR05": {}
}

def _add_common_parts_to_json_files(data, parking_lot_to_json_dict):
    for parking_lot in parking_lot_to_json_dict.keys():
        parking_lot_to_json_dict[parking_lot]["info"] = data["info"]
        parking_lot_to_json_dict[parking_lot]["licenses"] = data["licenses"]
        parking_lot_to_json_dict[parking_lot]["categories"] = data["categories"]
        parking_lot_to_json_dict[parking_lot]["images"] = []
        parking_lot_to_json_dict[parking_lot]["annotations"] = []

def _remove_old_files_and_create_paths(root_dir, parking_lot_to_json_file_path):
    import shutil
    for parking_lot in parking_lot_to_json_file_path.keys():
        path = os.path.join(root_dir, parking_lot)
        if os.path.exists(path):
            shutil.rmtree(path)
            print("Removed folder: ", path)
    for parking_lot in parking_lot_to_json_file_path.keys():
        train_path = os.path.join(root_dir, parking_lot, "train")
        valid_path = os.path.join(root_dir, parking_lot, "valid")
        test_path = os.path.join(root_dir, parking_lot, "test")
        os.makedirs(train_path)
        os.makedirs(valid_path)
        os.makedirs(test_path)
        print("Created folders: {}, {}, and {}".format(train_path, valid_path, test_path))

def split_by_parking_lot_main(root_dir):

    parking_lot_to_json_file_path = {
        "PUCPR": {
            "train": os.path.join(root_dir, "PUCPR/train/_annotations.coco.json"),
            "valid": os.path.join(root_dir, "PUCPR/valid/_annotations.coco.json"),
            "test": os.path.join(root_dir, "PUCPR/test/_annotations.coco.json")
        },
        "UFPR04": {
            "train": os.path.join(root_dir, "UFPR04/train/_annotations.coco.json"),
            "valid": os.path.join(root_dir, "UFPR04/valid/_annotations.coco.json"),
            "test": os.path.join(root_dir, "UFPR04/test/_annotations.coco.json")
        },
        "UFPR05": {
            "train": os.path.join(root_dir, "UFPR05/train/_annotations.coco.json"),
            "valid": os.path.join(root_dir, "UFPR05/valid/_annotations.coco.json"),
            "test": os.path.join(root_dir, "UFPR05/test/_annotations.coco.json")
        }
    }
    # Clean and remove old files if any exist
    _remove_old_files_and_create_paths(root_dir, parking_lot_to_json_file_path)
    for train_valid_test_selector, train_valid_test_path in zip(['train', 'valid', 'test'], [config.train_coco, config.val_coco, config.test_coco]):
        # Read the COCO annotations file
        with open(train_valid_test_path, "r") as f:
            data = json.load(f)
            num_parsed_imgs = 1
            _add_common_parts_to_json_files(data, parking_lot_to_json_dict)
            for image in data["images"]:
                date = image["file_name"][:10] # Get only the first 10 characters of the file name, which is the date in the format "YYYY-MM-DD"
                image_id = image["id"]
                # get the list of annotations that belong to this image
                annotations = [annotation for annotation in data["annotations"] if annotation["image_id"] == image_id]
                for parking_lot, dates in parking_lot_to_dates_dict.items():
                    if dates[0] <= date <= dates[1]:
                        parking_lot_to_json_dict[parking_lot]["images"].append(image)
                        parking_lot_to_json_dict[parking_lot]["annotations"].extend(annotations)
                        break
                print(f"Done processing {num_parsed_imgs} / {len(data['images'])} images for {train_valid_test_selector} data", end="\r")
                num_parsed_imgs += 1
        print(f"\nDone processing {train_valid_test_selector} COCO annotations file")
        # Write the new COCO annotations files
        for parking_lot, json_dict in parking_lot_to_json_dict.items():
            with open(parking_lot_to_json_file_path[parking_lot][train_valid_test_selector], "w") as f:
                json.dump(json_dict, f)
            print(f"Done writing {parking_lot} COCO annotations file to the path: {parking_lot_to_json_file_path[parking_lot][train_valid_test_selector]}")
        
def analyze_coco_file(coco_file):
    print("Analyzing COCO file: ", coco_file)
    with open(coco_file, "r") as f:
        data = json.load(f)
        print("Number of images: ", len(data["images"]))
        print("Number of annotations: ", len(data["annotations"]))
        empty_spaces = [annotation for annotation in data["annotations"] if annotation["category_id"] == 1]
        occupied_spaces = [annotation for annotation in data["annotations"] if annotation["category_id"] == 2]
        empty_percent = len(empty_spaces) / len(data["annotations"]) * 100
        print(f"Number of empty spaces: {len(empty_spaces)} ({empty_percent:.2f}%)")
        occupied_percent = len(occupied_spaces) / len(data["annotations"]) * 100
        print(f"Number of occupied spaces: {len(occupied_spaces)} ({occupied_percent:.2f}%)")

if __name__ == "__main__":
    split_by_parking_lot_main(config.SPLIT_BY_PARKING_LOT_ROOT_DIR)
    
    print("Overall dataset analysis")
    analyze_coco_file(config.train_coco)
    analyze_coco_file(config.val_coco)
    analyze_coco_file(config.test_coco)

    for parking_lot in ["PUCPR", "UFPR04", "UFPR05"]:
        print(f"{parking_lot} dataset analysis")
        analyze_coco_file(os.path.join(config.SPLIT_BY_PARKING_LOT_ROOT_DIR, f"{parking_lot}/train/_annotations.coco.json"))
        analyze_coco_file(os.path.join(config.SPLIT_BY_PARKING_LOT_ROOT_DIR, f"{parking_lot}/valid/_annotations.coco.json"))
        analyze_coco_file(os.path.join(config.SPLIT_BY_PARKING_LOT_ROOT_DIR, f"{parking_lot}/test/_annotations.coco.json"))
    