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
import matplotlib.pyplot as plt
import numpy as np

def visualize_method_per_row(structure_dict):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(dir_path, 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    import pickle
    data_dict = {}
    for method in structure_dict:
        data_dict[method] = {}
        for local_epochs_key_str in structure_dict[method].keys():
            data_dict[method][local_epochs_key_str] = {}
            for client_name in ['site-1', 'site-2', 'site-3', 'site-4']:
                one_epoch_pkl_path = os.path.join(EXPERIMENTS_ROOT_DIR, structure_dict[method][local_epochs_key_str], client_name, 'overall_trackers.pkl')
                with open(one_epoch_pkl_path, "rb") as f:
                    data_dict[method][local_epochs_key_str][client_name] = pickle.load(f)
    print("Finished reading the pickle files.")
    method = 'FedAvg'
    client_name = 'site-1'
    # Visualize_method_per_row the losses
    #Create a figure and 4x3 subplots
    fig, axs = plt.subplots(3, 4)
    # Set the size of the figure to the width of a letter size paper
    fig.set_size_inches(12, 8.5)

    y_limits_per_row = []
    for i, method in enumerate(structure_dict):
        for j, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
            max_loss = 0
            for local_epochs_key_str in structure_dict[method].keys():
                max_loss = max(max_loss, max(data_dict[method][local_epochs_key_str][client_name]['train_loss']))
            y_limits_per_row.append(max_loss)

    # fig.suptitle('Client 1 - FedAvg')
    for i, method in enumerate(structure_dict):
        for j, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
            for local_epochs_key_str in structure_dict[method].keys():
                num_epochs = len(data_dict[method][local_epochs_key_str][client_name]['train_loss'])
                epochs_arr = np.arange(num_epochs)
                axs[i, j].plot(epochs_arr, data_dict[method][local_epochs_key_str][client_name]['train_loss'], label=f'{local_epochs_key_str}')
            if i == 0:
                axs[i, j].set(title=f'Client {j+1}')
            if j == 0:
                axs[i, j].set(ylabel=f"{method} Loss")
            else:
                # Remove the y-axis numbers from all subplots except the first one
                axs[i, j].yaxis.set_tick_params(labelleft=False)

            if i == len(structure_dict) - 1:
                axs[i, j].set(xlabel="Epoch")
           
            # Set y-axis limits
            axs[i, j].set_ylim([0, y_limits_per_row[i]]) 

            # Display legends in the last column
            if j == 3: # assuming that we have only 4 clients
                axs[i, j].legend()

            axs[i, j].grid()

    # fig, axs = plt.subplots(4, 3)
    # ax.plot(epochs_arr, data_dict[method]['1-local-epoch'][client_name]['train_loss'], label='1 local epoch')
    # ax.plot(epochs_arr, data_dict[method]['2-local-epochs'][client_name]['train_loss'], label='2 local epochs')
    # ax.set(xlabel="Epoch", ylabel="Loss", title="Client 1 - FedAvg")
    # ax.grid()
    plt.show()

    # Save the plot to a PDF file
    losses_img_path = os.path.join(output_dir, 'fl-losses.pdf')
    fig.savefig(losses_img_path, bbox_inches='tight')
    print(f"Losses plot saved to {losses_img_path}")

    # Draw mAP vs Rounds
    fig, axs = plt.subplots(3, 4)
    # Set the size of the figure to the width of a letter size paper
    fig.set_size_inches(12, 8.5)

    y_limits_per_row = []
    for i, method in enumerate(structure_dict):
        for j, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
            max_loss = 0
            for local_epochs_key_str in structure_dict[method].keys():
                max_loss = max(max_loss, max(data_dict[method][local_epochs_key_str][client_name]['train_loss']))
            y_limits_per_row.append(max_loss)

    # fig.suptitle('Client 1 - FedAvg')
    for i, method in enumerate(structure_dict):
        for j, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
            for local_epochs_key_str in structure_dict[method].keys():
                mAP = [x['mAP'] for x in data_dict[method][local_epochs_key_str][client_name]['val_acc']]
                rounds_arr = np.arange(len(mAP))
                axs[i, j].plot(rounds_arr, mAP, label=f'{local_epochs_key_str}')
            if i == 0:
                axs[i, j].set(title=f'Client {j+1}')
            if j == 0:
                axs[i, j].set(ylabel=f"{method} mAP")
            else:
                # Remove the y-axis numbers from all subplots except the first one
                axs[i, j].yaxis.set_tick_params(labelleft=False)

            if i == len(structure_dict) - 1:
                axs[i, j].set(xlabel="Round")
           
            # Set y-axis limits
            axs[i, j].set_ylim([0, 1.1]) 

            # Display legends in the last column
            if j == 3: # assuming that we have only 4 clients
                axs[i, j].legend(loc='lower right')

            axs[i, j].grid()

    # fig, axs = plt.subplots(4, 3)
    # ax.plot(epochs_arr, data_dict[method]['1-local-epoch'][client_name]['train_loss'], label='1 local epoch')
    # ax.plot(epochs_arr, data_dict[method]['2-local-epochs'][client_name]['train_loss'], label='2 local epochs')
    # ax.set(xlabel="Epoch", ylabel="Loss", title="Client 1 - FedAvg")
    # ax.grid()
    plt.show()

    # Save the plot to a PDF file
    mAP_img_path = os.path.join(output_dir, 'fl-mAP.pdf')
    fig.savefig(mAP_img_path, bbox_inches='tight')
    print(f"mAP plot saved to {mAP_img_path}")

    
def visualize_clients_per_row(structure_dict):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(dir_path, 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    import pickle
    data_dict = {}
    for method in structure_dict:
        data_dict[method] = {}
        for local_epochs_key_str in structure_dict[method].keys():
            data_dict[method][local_epochs_key_str] = {}
            for client_name in ['site-1', 'site-2', 'site-3', 'site-4']:
                one_epoch_pkl_path = os.path.join(EXPERIMENTS_ROOT_DIR, structure_dict[method][local_epochs_key_str], client_name, 'overall_trackers.pkl')
                with open(one_epoch_pkl_path, "rb") as f:
                    data_dict[method][local_epochs_key_str][client_name] = pickle.load(f)
    print("Finished reading the pickle files.")
    method = 'FedAvg'
    client_name = 'site-1'
    # Visualize_method_per_row the losses
    #Create a figure and 4x3 subplots
    fig, axs = plt.subplots(4, 3)
    # Set the size of the figure to the width of a letter size paper
    fig.set_size_inches(4.5, 6.8)

    y_limits_per_row = []
    for i, method in enumerate(structure_dict):
        for j, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
            max_loss = 0
            for local_epochs_key_str in structure_dict[method].keys():
                max_loss = max(max_loss, max(data_dict[method][local_epochs_key_str][client_name]['train_loss']))
            y_limits_per_row.append(max_loss)

    handles, labels = [], []
    for i, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
        for j, method in enumerate(structure_dict):
            for local_epochs_key_str in structure_dict[method].keys():
                num_epochs = len(data_dict[method][local_epochs_key_str][client_name]['train_loss'])
                epochs_arr = np.arange(num_epochs)
                line, = axs[i, j].plot(epochs_arr, data_dict[method][local_epochs_key_str][client_name]['train_loss'])
                if len(handles) < 2:
                    handles.append(line)
                    labels.append(f'{local_epochs_key_str}'.replace('-', ' '))
            if i == 0:
                # Set the title of the first row and change the fontsize to 9
                axs[i, j].set_title(f'{method}', fontsize='9')
            if j == 0:
                axs[i, j].set_ylabel(f"Client {i+1} Loss", fontsize='9')
            else:
                # Remove the y-axis numbers from all subplots except the first one
                axs[i, j].yaxis.set_tick_params(labelleft=False)

            if i == 3: # assuming that we have only 4 clients
                axs[i, j].set_xlabel("Epoch", fontsize='9')
           
            # Set y-axis limits
            axs[i, j].set_ylim([0, y_limits_per_row[i]]) 

            axs[i, j].grid()
    # Create a single legend for all subplots and change the font size
    fig.legend(handles, labels, loc='upper left', fontsize='9')
    plt.show()

    # Save the plot to a PDF file
    losses_img_path = os.path.join(output_dir, 'fl_losses.pdf')
    fig.savefig(losses_img_path, bbox_inches='tight')
    print(f"Losses plot saved to {losses_img_path}")

    # Draw mAP vs Epoch
    #Create a figure and 4x3 subplots
    fig, axs = plt.subplots(4, 3)
    # Set the size of the figure to the width of a letter size paper
    fig.set_size_inches(4.5, 6.8)
    handles, labels = [], []
    for i, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
        for j, method in enumerate(structure_dict):
            for local_epochs_key_str in structure_dict[method].keys():
                mAP = [x['mAP'] for x in data_dict[method][local_epochs_key_str][client_name]['val_acc']]
                rounds_arr = np.arange(len(mAP))
                line, = axs[i, j].plot(rounds_arr, mAP)
                if len(handles) < 2:
                    handles.append(line)
                    labels.append(f'{local_epochs_key_str}'.replace('-', ' '))
            if i == 0:
                # Set the title of the first row and change the fontsize to 9
                axs[i, j].set_title(f'{method}', fontsize='9')
            if j == 0:
                axs[i, j].set_ylabel(f"Client {i+1} mAP", fontsize='9')
            else:
                # Remove the y-axis numbers from all subplots except the first one
                axs[i, j].yaxis.set_tick_params(labelleft=False)

            if i == 3: # assuming that we have only 4 clients
                axs[i, j].set_xlabel("Epoch", fontsize='9')
           
            # Set y-axis limits
            axs[i, j].set_ylim([0.25, 1.1]) 

            axs[i, j].grid()
    # Create a single legend for all subplots and change the font size
    fig.legend(handles, labels, loc='upper left', fontsize='9')
    plt.show()

    # Save the plot to a PDF file
    mAP_img_path = os.path.join(output_dir, 'fl_mAP.pdf')
    fig.savefig(mAP_img_path, bbox_inches='tight')
    print(f"mAP plot saved to {mAP_img_path}")
            


EXPERIMENTS_ROOT_DIR = "/home/bakr/NVFlare/examples/hello-world/parking-federated-training/exprs"
structure_dict = {
    'FedAvg': {
        '1-local-epoch': 'expr-10',
        '2-local-epochs': 'expr-09'
    },
    'FedProx': {
        '1-local-epoch': 'expr-11',
        '2-local-epochs': 'expr-12'
    },
    'SCAFFOLD': {
        '1-local-epoch': 'expr-13',
        '2-local-epochs': 'expr-14'
    }
}

visualize_method_per_row(structure_dict)
# visualize_clients_per_row(structure_dict)
