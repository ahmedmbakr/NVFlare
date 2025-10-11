# The structure of the file that has the experiments results is as follows:
# expr-xx:
#   - site-1
#     - overall_trackers.pkl
#   - site-2
#     - overall_trackers.pkl
#   - site-3
#     - overall_trackers.pkl
#   - site-4
#     - overall_trackers.pkl

import os

def convert_into_csv(structure_dict):
    import pickle
    for method in structure_dict:
        for local_epochs_key_str in structure_dict[method].keys():
            data_dict = {}
            for client_name in ['site-1', 'site-2', 'site-3', 'site-4']:
                one_epoch_pkl_path = os.path.join(EXPERIMENTS_ROOT_DIR, structure_dict[method][local_epochs_key_str], client_name, 'overall_trackers.pkl')
                with open(one_epoch_pkl_path, "rb") as f:
                    data_dict[client_name] = pickle.load(f)

            # Create a CSV file for the one epoch experiment
            current_file_dir = os.path.dirname(os.path.realpath(__file__))
            outputs_dir_path = os.path.join(current_file_dir, 'outputs')
            if not os.path.exists(outputs_dir_path):
                os.makedirs(outputs_dir_path)
            one_local_epoch_csv_file_path = os.path.join(outputs_dir_path, f'{method}_{local_epochs_key_str}.csv')
            with open(one_local_epoch_csv_file_path, 'w') as f:
                f.write('round,Client-1,Client-2,Client-3,Client-4\n')
                for i, (loss1, loss2, loss3, loss4) in enumerate(zip(data_dict['site-1']['train_loss'], data_dict['site-2']['train_loss'], data_dict['site-3']['train_loss'], data_dict['site-4']['train_loss'])):
                    f.write(f"{i},{loss1},{loss2},{loss3},{loss4}\n")
            print(f"CSV file for {method} {local_epochs_key_str} experiment is created at {one_local_epoch_csv_file_path}")

def create_combined_csv_files(structure_dict):
    import pandas as pd
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    outputs_dir_path = os.path.join(current_file_dir, 'outputs')
    for method in structure_dict:
        one_local_epoch_csv_file_path = os.path.join(outputs_dir_path, f'{method}_1-local-epoch.csv')
        two_local_epochs_csv_file_path = os.path.join(outputs_dir_path, f'{method}_2-local-epochs.csv')
        one_local_epoch_df = pd.read_csv(one_local_epoch_csv_file_path)
        two_local_epochs_df = pd.read_csv(two_local_epochs_csv_file_path)
        combined_df = pd.concat([one_local_epoch_df, two_local_epochs_df], axis=1)
        combined_csv_file_path = os.path.join(outputs_dir_path, f'{method}_combined.csv')
        combined_df.to_csv(combined_csv_file_path, index=False)
        print(f"Combined CSV file for {method} is created at {combined_csv_file_path}")


EXPERIMENTS_ROOT_DIR = "/home/bakr/NVFlare/examples/hello-world/parking-federated-training/exprs"
structure_dict = {
    'FedAvg': {
        '1-local-epoch': 'expr-10',
        '2-local-epochs': 'expr-09'
    },
    'FedProx': {
        '1-local-epoch': 'expr-11',
        '2-local-epochs': 'expr-11' # TODO: This should be expr-12, but the file is corrupt.
    },
    'SCAFFOLD': {
        '1-local-epoch': 'expr-13',
        '2-local-epochs': 'expr-14'
    }
}

convert_into_csv(structure_dict)
create_combined_csv_files(structure_dict)
