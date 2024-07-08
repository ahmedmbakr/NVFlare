import numpy as np
import torch
import os, sys
import torch.utils.data

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '../jobs/parking-federated-training/app/custom')))
print(sys.path)
import Resnet
import parkingFL_Tester
import PkLotDataLoader

def validate_on_test_data(poc_workspace: str, models_full_paths_list: list, models_names, num_clients: int, test_coco_full_path_pattern: str, outputs_dir: str, valid_detection_threshold: int, batch_size: int, num_workers_dl: int, num_classes=3):
    validation_results = np.zeros((num_clients + 1, num_clients))
    model = Resnet.ResnetFasterRCNN.get_pretrained_model(num_classes)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    checkpoint = torch.load(models_full_paths_list[-1])
    model.load_state_dict(checkpoint ['model'])
    model.to(device)

    for idx, (model_owner, model_full_path) in enumerate(zip(models_names, models_full_paths_list)):
        # The saved file is a dictionary with the model and the best epoch number
        model_dict_key = "model" if model_owner == "server" else "model_weights"
        checkpoint  = torch.load(model_full_path)
        model.load_state_dict(checkpoint [model_dict_key])
        model.to(device)
        for client_idx in range(num_clients): # We test the model on client_idx data
             
            test_coco_full_path = test_coco_full_path_pattern.format(client_idx + 1)
            # Get the directory of the test data
            test_data_dir = os.path.dirname(test_coco_full_path)
            test_dataset = PkLotDataLoader.PklotDataSet(
            root_path=test_data_dir, annotation_path=test_coco_full_path, transforms=Resnet.ResnetFasterRCNN.get_transform()
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers_dl,
                collate_fn=PkLotDataLoader.collate_fn,
                )
        
            mAP = parkingFL_Tester.ParkingFL_Tester.validate_model_on_test_data(model, model_owner, outputs_dir, device, test_loader, valid_detection_threshold)
            validation_results[idx, client_idx] = mAP
            print("mAP for model {} on client {} data is: {}".format(model_owner, client_idx + 1, mAP))
    # Save the results in a csv file.
    np.savetxt(os.path.join(outputs_dir, "validation_results.csv"), validation_results, delimiter=",")
    # Print the results in a table format
    print("Final Validation Results:")
    print(validation_results)
    return validation_results
        
        
if __name__ == "__main__":
    POC_WORKSPACE = "/tmp/bakr-nvflare/poc/example_project/prod_00"
    task_id = "16e9e73b-c262-4580-8482-f35fe4ebf6a2"
    models_full_paths_list = [
        f'{POC_WORKSPACE}/site-1/{task_id}/app_site-1/outputs/models/model_99_0_99.pth', # Client 1
        f'{POC_WORKSPACE}/site-2/{task_id}/app_site-2/outputs/models/model_99_0_99.pth', # Client 2
        f'{POC_WORKSPACE}/site-3/{task_id}/app_site-3/outputs/models/model_99_0_99.pth', # Client 3
        f'{POC_WORKSPACE}/site-4/{task_id}/app_site-4/outputs/models/model_99_0_99.pth', # Client 4
        f'{POC_WORKSPACE}/FL_global_model.pt' # Server
        ]
    models_names = ['site-1', 'site-2', 'site-3', 'site-4', 'server']
    test_coco_full_path_pattern = POC_WORKSPACE + "/site-{}/data/test/_annotations.coco.json"
    num_clients = 4
    outputs_dir = os.path.abspath(os.path.join(POC_WORKSPACE, 'test_data_outputs'))
    valid_detection_threshold = 0.5
    batch_size = 6
    num_workers = 4
    _ = validate_on_test_data(POC_WORKSPACE, models_full_paths_list, models_names, num_clients, test_coco_full_path_pattern, outputs_dir, valid_detection_threshold, batch_size, num_workers)
