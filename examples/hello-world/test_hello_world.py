import os
import time

from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.flare_api.flare_api import NoConnection 

import json
from nvflare.fuel.flare_api.flare_api import Session

# Before running, change the working directory to : examples/hello-world

def status_monitor_cb(
        session: Session, job_id: str, job_meta, *cb_args, **cb_kwargs
    ) -> bool:
    if job_meta["status"] == "RUNNING":
        if cb_kwargs["cb_run_counter"]["count"] < 3 or cb_kwargs["cb_run_counter"]["count"]%15 == 0:
            print(job_meta)            
        else:
            # avoid printing job_meta repeatedly to save space on the screen and not overwhelm the user
            print(".", end="")
    else:
        print("\n" + str(job_meta))
    
    cb_kwargs["cb_run_counter"]["count"] += 1
    return True


def format_json( data: dict):
    # Helper function to format output of list_jobs()
    print(json.dumps(data, sort_keys=True, indent=4,separators=(',', ': ')))

workspace = "/tmp/nvflare/poc"
default_poc_prepared_dir = os.path.join(workspace, "example_project/prod_00")
admin_dir = os.path.join(default_poc_prepared_dir, "admin@nvidia.com")

# the following try/except is usually not needed, we need it here to handle the case when you "Run all cells" or use notebook automation. 
# in the "Run all cells" case, JupyterLab seems to try to connect to the server before it starts (even though the execution is supposed to be sequential),
# which will result in a connection timeout. We use try/except to capture the scenario since extra sleep time doesn't seem to help.

# try: 
#    sess = new_secure_session("admin@nvidia.com", admin_dir, timeout=5)
# except NoConnection:
#     time.sleep(10)
    
    
# flare_not_ready = True
# while flare_not_ready: 
#     print("trying to connect to server")
#     try:
#         sess = new_secure_session("admin@nvidia.com", admin_dir)
#     except NoConnection:
#         print("CANNOT CONNECT AFTER 10 SECONDS")
#         continue

#     sys_info = sess.get_system_info()

#     print(f"Server info:\n{sys_info.server_info}")
#     print("\nClient info")
#     for client in sys_info.client_info:
#         print(client)
#     flare_not_ready = len( sys_info.client_info) < 2
        
#     time.sleep(2)

import os
from nvflare.fuel.flare_api.flare_api import new_secure_session

poc_workspace = "/tmp/nvflare/poc"
poc_prepared = os.path.join(poc_workspace, "example_project/prod_00")
admin_dir = os.path.join(poc_prepared, "admin@nvidia.com")
sess = new_secure_session("admin@nvidia.com", startup_kit_location=admin_dir)

job_folder = os.path.join(os.getcwd(), "hello-numpy-sag/jobs/hello-numpy-sag")
job_id = sess.submit_job(job_folder)

print(f"Job is running with ID {job_id}")

# Wait for the job.
sess.monitor_job(job_id, cb=status_monitor_cb, cb_run_counter={"count":0})

list_jobs_output_detailed = sess.list_jobs(detailed=True)
print(format_json(list_jobs_output_detailed))

sess.get_job_meta(job_id)

import numpy as np
# job_id = 'ce198d57-a552-4458-8769-10cd0ae5dade'
print(job_id)
result = sess.download_job_result(job_id)
print(result)
array = np.load(result + "/workspace/models/server.npy")
print(array)
