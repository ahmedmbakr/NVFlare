"""
This is the admin automation script that submits the job, monitors its progress, and closes the session once the execution is done.
Developer: Ahmed Bakr
Date: 2024-06-09
"""

from nvflare.fuel.flare_api.flare_api import *
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
    

if __name__ == "__main__":
    POC_WORKSPACE = "/tmp/nvflare/poc/example_project/prod_00"
    ADMIN_NAME = "admin@nvidia.com"
    JOB_NAME = "parking-federated-training"
    POLL_STATUS_INTERVAL_SEC = 2

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
        # Execute shell command
        os.system("nvflare poc stop")
        print("System is shut down")
    finally:
        sess.close()
