"""
This is the admin automation script that submits the job, monitors its progress, and closes the session once the execution is done.
Developer: Ahmed Bakr
Date: 2024-06-09
"""

from nvflare.fuel.flare_api.flare_api import *
from utils.FL_validate_on_test_data import validate_on_test_data
import os

"""
This callback function is used to monitor the job progress. It is the callback of the monitor_job function.
It is exeucted every POLL_STATUS_INTERVAL_SEC seconds.
"""
def sample_cb(
    session: Session, job_id: str, job_meta, *cb_args, **cb_kwargs
) -> bool:
    if job_meta["status"] == "RUNNING":
        if cb_kwargs["cb_run_counter"]["count"] < 3:
            print(job_meta)
            print(cb_kwargs["cb_run_counter"])
        else:
            # print(".", end="")
            print(os.popen("nvidia-smi").read())
    else:
        print("\n" + str(job_meta))

    cb_kwargs["cb_run_counter"]["count"] += 1
    return True

def visualize_graphs(num_clients, job_id, trackers_file_path_pattern):
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt

    for i in range(1, num_clients + 1):
        trackers_file_path = trackers_file_path_pattern.format(i, job_id)
        vis_figures_folder_path = os.path.dirname(trackers_file_path) + "/visualizations"
        if os.path.exists(vis_figures_folder_path):
            import shutil
            shutil.rmtree(vis_figures_folder_path)
        os.makedirs(vis_figures_folder_path)
        from utils.visualize_exp_pkl import visualize_pkl_file
        visualize_pkl_file(trackers_file_path, vis_figures_folder_path)
        print("Visualizations for site{} are saved to the path: {}".format(i, vis_figures_folder_path))

def test_models_on_test_data(poc_workspace, job_id, num_clients, model_name, valid_detection_threshold=0.5, batch_size=6, num_workers=4):
    models_full_paths_list = []
    models_names = []
    for i in range(1, num_clients + 1):
        models_full_paths_list.append(f'{poc_workspace}/site-{i}/{job_id}/app_site-{i}/outputs/models/local_model.pt')
        models_names.append(f'site-{i}')

    models_full_paths_list.append(f'{poc_workspace}/FL_global_model.pt') # Server' model
    models_names.append('server')

    test_coco_full_path_pattern = poc_workspace + "/site-{}/data/test/_annotations.coco.json"
    outputs_dir = os.path.abspath(os.path.join(poc_workspace, 'test_data_outputs'))
    _ = validate_on_test_data(poc_workspace, models_full_paths_list, models_names, num_clients, test_coco_full_path_pattern, outputs_dir, valid_detection_threshold, batch_size, num_workers, model_name)

if __name__ == "__main__":
    POC_WORKSPACE = "/tmp/bakr-nvflare/poc/example_project/prod_00"
    ADMIN_NAME = "admin@nvidia.com"
    JOB_NAME = "parking-federated-training"
    POLL_STATUS_INTERVAL_SEC = 2
    NUM_CLIENTS = 4
    TRACKERS_FILE_PATH = POC_WORKSPACE + "/site-{0}/{1}/app_site-{0}/outputs/overall_trackers.pkl" # AB: Param0: site number, Param1: job_id
    MODEL_NAME = "yolov5"

    import time
    # print("Waiting for 20 seconds to give the server time to start")
    # # Sleep for 20 seconds to give the server time to start
    # time.sleep(20) # Give the server time to start after running the command "nvflare poc start"
    # print("Done waiting for 20 seconds. Starting the script...")
    admin_path = os.path.join(POC_WORKSPACE, ADMIN_NAME)
    print("Admin Path: ", admin_path)
    sess = new_secure_session(
        ADMIN_NAME,
        admin_path
    )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        print(sess.get_system_info())
        job_full_path = os.path.join(dir_path, "jobs", JOB_NAME)
        job_id = sess.submit_job(job_full_path)
        print(job_id + " was submitted")
        sess.monitor_job(job_id, poll_interval=POLL_STATUS_INTERVAL_SEC, cb=sample_cb, cb_run_counter={"count":0})
        print("job done!")

        # Save the Server's global model and logs
        server_global_model_path = f'{POC_WORKSPACE}/server/{job_id}/app_server/FL_global_model.pt'
        os.system(f"cp {server_global_model_path} {POC_WORKSPACE}")
        print("Saved the server's global model to the path: ", f"{POC_WORKSPACE}/FL_global_model.pt")

        server_logs_path = f'{POC_WORKSPACE}/server/{job_id}/log.txt'
        os.system(f"cp {server_logs_path} {POC_WORKSPACE}/server_logs.txt")
        print("saved the server logs to the path: ", f"{POC_WORKSPACE}/server_logs.txt")

        visualize_graphs(NUM_CLIENTS, job_id, TRACKERS_FILE_PATH)

        test_models_on_test_data(POC_WORKSPACE, job_id, NUM_CLIENTS, MODEL_NAME)
        # Original command: zip -r ~/NVFlare/examples/hello-world/parking-federated-training/poc_output.zip /tmp/bakr-nvflare/poc/example_project/prod_00
        zip_file_path = os.path.join(dir_path, "poc_output.zip")
        os.system(f"zip -r {zip_file_path} {POC_WORKSPACE}/..")
        print("POC output is zipped to the path: ", zip_file_path)

    finally:
        sess.close()
        # Execute shell command
        os.system("nvflare poc stop")
        print("System is shut down")
