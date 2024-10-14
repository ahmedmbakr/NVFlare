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

def calculate_f1_score_iou_50(data_dict, structure_dict):
    # Loop through each network, method, and client to calculate mean F1 score for IoU 50%
    for network_name in structure_dict:
        network_structure_dict = structure_dict[network_name]
        for method in network_structure_dict:
            for client_name in ['site-1', 'site-2', 'site-3', 'site-4']:
                f1_scores = []
                for local_epochs_key_str in data_dict[network_name][method]:
                    # Determine final round based on local epoch strategy
                    final_round_index = -1 if '1-local-epoch' in local_epochs_key_str else -1 // 2
                    
                    # Fetch the metrics for the final round
                    val_acc = data_dict[network_name][method][local_epochs_key_str][client_name]['val_acc']
                    if val_acc:
                        last_epoch_data = val_acc[final_round_index]
                        precision_per_class = last_epoch_data.get('precision', {})
                        recall_per_class = last_epoch_data.get('recall', {})
                        
                        # Calculate F1 score for each class at IoU 50%
                        for class_name in precision_per_class:
                            precisions = np.array(precision_per_class.get(class_name, []))
                            recalls = np.array(recall_per_class.get(class_name, []))
                            
                            # Assuming IoU 50% is at the first index (index 0)
                            if len(precisions) > 0 and len(recalls) > 0:
                                precision_iou_50 = precisions[0]
                                recall_iou_50 = recalls[0]
                                
                                # Calculate F1 score for IoU 50%
                                if (precision_iou_50 + recall_iou_50) > 0:
                                    f1_score = 2 * (precision_iou_50 * recall_iou_50) / (precision_iou_50 + recall_iou_50)
                                else:
                                    f1_score = 0
                                
                                f1_scores.append(f1_score)
                                print(f"F1 Score for {network_name} - {method} - {client_name} - {class_name} (IoU 50%): {f1_score:.4f}")
                
                # Calculate mean F1 score for this method and client across all classes
                if f1_scores:
                    mean_f1_score = np.mean(f1_scores)
                    print(f"Mean F1 Score for {network_name} - {method} - {client_name} (IoU 50%): {mean_f1_score:.4f}\n")
                else:
                    print(f"No F1 Score data for {network_name} - {method} - {client_name}")


def visualize_final_round_f1_score(data_dict, structure_dict, output_dir):
    # Determine the total number of rows based on the combinations of network_name and method
    total_rows = sum(len(methods) for methods in structure_dict.values())
    num_clients = 4  # Number of clients

    # Create a figure and subplots
    fig, axs = plt.subplots(total_rows, num_clients, figsize=(12.1, 8.5))

    # Flatten axs for ease of indexing if total_rows == 1
    if total_rows == 1:
        axs = [axs]

    # Extract class names from data_dict
    class_names = None
    for network_name in data_dict:
        for method in data_dict[network_name]:
            for local_epochs_key_str in data_dict[network_name][method]:
                for client_name in data_dict[network_name][method][local_epochs_key_str]:
                    val_acc = data_dict[network_name][method][local_epochs_key_str][client_name]['val_acc']
                    if val_acc:
                        last_epoch_data = val_acc[-1]
                        precision_per_class = last_epoch_data.get('precision', {})
                        class_names = list(precision_per_class.keys())
                        break
                if class_names:
                    break
            if class_names:
                break
        if class_names:
            break
    if not class_names:
        print("No class names found in data.")
        return

    # Initialize row index
    i = 0
    for network_name in structure_dict:
        network_structure_dict = structure_dict[network_name]
        for method in network_structure_dict:
            for j, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
                f1_scores = {class_name: {} for class_name in class_names}
                
                for local_epochs_key_str in data_dict[network_name][method]:
                    # Determine final round based on local epoch strategy
                    final_round_index = -1
                    
                    # Fetch the metrics for the final round
                    val_acc = data_dict[network_name][method][local_epochs_key_str][client_name]['val_acc']
                    if val_acc:
                        last_epoch_data = val_acc[final_round_index]
                        precision_per_class = last_epoch_data.get('precision', {})
                        recall_per_class = last_epoch_data.get('recall', {})
                        
                        # Calculate F1 score for each class
                        for class_name in class_names:
                            precision = precision_per_class.get(class_name, 0)
                            recall = recall_per_class.get(class_name, 0)
                            f1_score = [  2 * (precision_elem * recall_elem) / (precision_elem + recall_elem) if  (precision_elem + recall_elem) > 0 else 0.0 for precision_elem, recall_elem in zip(precision, recall)]
                            f1_scores[class_name][local_epochs_key_str] = f1_score
                
                # Plotting
                for class_name, color in zip(class_names, ['r', 'g']):
                    for local_epochs_key_str in data_dict[network_name][method]:
                        # if local_epochs_key_str not in f1_scores[class_name]:
                        #     continue
                        x_values = range(len(f1_scores[class_name][local_epochs_key_str]))
                        line_style = '-' if '1-local-epoch' in local_epochs_key_str else '--'
                        class_name_txt = 'Empty' if class_name == 'Space-empty' else 'Occupied'
                        axs[i][j].plot(x_values, f1_scores[class_name][local_epochs_key_str], f"{color}{line_style}", label=f"{class_name_txt}-{local_epochs_key_str}")
                    
                axs[i][j].set_ylim([0, 1])
                if i == 0:
                    axs[i][j].set(title=f'Client {j+1}')
                if j == 0:
                    axs[i][j].set(ylabel=f"{method}\nfor {network_name}")
                else:
                    axs[i][j].yaxis.set_tick_params(labelleft=False)
                if i == total_rows - 1:
                    axs[i][j].set(xlabel='Round')
                axs[i][j].grid(True)
                
                # Show legend in the last column
                if j == num_clients - 1:
                    axs[i][j].legend(fontsize=9)
            i += 1

    plt.tight_layout()
    # Save the plot to a PDF file
    f1_img_path = os.path.join(output_dir, 'f1_score_per_class_final_round.pdf')
    fig.savefig(f1_img_path, bbox_inches='tight')
    print(f"F1 Score per class plot saved to {f1_img_path}")
    plt.show()

def visualize_final_round_precision_recall(data_dict, structure_dict, output_dir):
    # Determine the total number of rows based on the combinations of network_name and method
    total_rows = sum(len(methods) for methods in structure_dict.values())
    num_clients = 4  # Assuming 4 clients

    # Create a figure and subplots
    fig, axs = plt.subplots(total_rows, num_clients, figsize=(12.1, 9))

    # Flatten axs for ease of indexing if total_rows == 1
    if total_rows == 1:
        axs = [axs]

    # Initialize row index
    row = 0
    for network_name, methods in structure_dict.items():
        for method in methods:
            for col, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
                # Initialize variables for final precision and recall values
                final_precision_1_epoch = {}
                final_recall_1_epoch = {}
                final_precision_2_epoch = {}
                final_recall_2_epoch = {}

                # Gather data for each setting
                for local_epochs_key_str in data_dict[network_name][method]:
                    val_acc = data_dict[network_name][method][local_epochs_key_str][client_name].get('val_acc', [])
                    
                    if local_epochs_key_str == '1-local-epoch':
                        # Round 99 for 1-local-epoch
                        idx = 99 if len(val_acc) > 99 else -1
                        metrics_per_class = val_acc[idx]
                        final_precision_1_epoch = metrics_per_class.get('precision', {})
                        final_recall_1_epoch = metrics_per_class.get('recall', {})
                    elif local_epochs_key_str == '2-local-epochs':
                        # Round 49 for 2-local-epochs
                        idx = 49 if len(val_acc) > 49 else -1
                        metrics_per_class = val_acc[idx]
                        final_precision_2_epoch = metrics_per_class.get('precision', {})
                        final_recall_2_epoch = metrics_per_class.get('recall', {})

                # Plot the precision-recall curve for 'Space-empty' and 'Space-occupied' classes
                for class_name, color in zip(['Space-empty', 'Space-occupied'], ['r', 'g']):
                    if final_recall_1_epoch and final_precision_1_epoch:
                        recall_1_epoch = final_recall_1_epoch.get(class_name, 0)
                        precision_1_epoch = final_precision_1_epoch.get(class_name, 0)
                        txt_class_name = 'Empty-1-local-epoch' if class_name == 'Space-empty' else 'Occupied-1-local-epoch'
                        axs[row][col].plot(recall_1_epoch, precision_1_epoch, f'{color}-', label=f'{txt_class_name}')

                    if final_recall_2_epoch and final_precision_2_epoch:
                        recall_2_epoch = final_recall_2_epoch.get(class_name, 0)
                        precision_2_epoch = final_precision_2_epoch.get(class_name, 0)
                        txt_class_name = 'Empty-2-local-epochs' if class_name == 'Space-empty' else 'Occupied-2-local-epochs'
                        axs[row][col].plot(recall_2_epoch, precision_2_epoch, f'{color}--', label=f'{txt_class_name}')

                # Labeling
                # axs[row][col].set_xlabel('Recall', fontsize=9)
                # axs[row][col].set_ylabel('Precision', fontsize=9)
                axs[row][col].set_xlim([-0.1, 1.05])
                axs[row][col].set_ylim([-0.1, 1.1])
                
                if row == 0:
                    axs[row][col].set_title(f'Client {col + 1}', fontsize=9)
                if col == 0:
                    axs[row][col].set_ylabel(f"{method} Precision\nfor {network_name}", fontsize=9)
                if row == total_rows - 1:
                    axs[row][col].set_xlabel('Recall', fontsize=9)
                # Enable grid and set legend
                axs[row][col].grid(True, linestyle='--', linewidth=0.5)
                axs[row][col].legend(loc='lower left', fontsize=9)
                
                # Set tick label size
                axs[row][col].tick_params(axis='both', labelsize=9)
            
            # Move to the next row for each method
            row += 1

    plt.tight_layout()

    # Save the plot to a PDF file
    pr_curve_img_path = os.path.join(output_dir, 'final_round_precision_recall_curve.pdf')
    fig.savefig(pr_curve_img_path, bbox_inches='tight')
    print(f"Final round precision-recall curve plot saved to {pr_curve_img_path}")
    plt.show()


def visualize_log_average_miss_rate(data_dict, structure_dict, output_dir):
    # Set font size to 9 for all plot elements
    # Determine the total number of rows based on the combinations of network_name and method
    total_rows = sum(len(methods) for methods in structure_dict.values())
    num_clients = 4  # Assuming 4 clients

    # Create a figure and subplots
    fig, axs = plt.subplots(total_rows, num_clients, figsize=(12.1, 9))

    # Flatten axs for ease of indexing if total_rows == 1
    if total_rows == 1:
        axs = [axs]

    # Initialize row index
    row = 0
    for network_name, methods in structure_dict.items():
        for method in methods:
            for col, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
                # Initialize lists to hold log-average miss rate values for each class and local epoch setting
                rounds_1_epoch, log_miss_empty_1_epoch, log_miss_occupied_1_epoch = [], [], []
                rounds_2_epoch, log_miss_empty_2_epoch, log_miss_occupied_2_epoch = [], [], []

                # Gather data for each setting
                for local_epochs_key_str in data_dict[network_name][method]:
                    val_acc = data_dict[network_name][method][local_epochs_key_str][client_name].get('val_acc', [])
                    
                    for round_num, epoch_data in enumerate(val_acc):
                        miss_rate_per_class = epoch_data.get('log_avg_miss_rate', {})
                        log_miss_empty = miss_rate_per_class.get('Space-empty', 1)  # Default to log(1e-8) if not found
                        log_miss_occupied = miss_rate_per_class.get('Space-occupied', 1)

                        # Separate data based on the number of local epochs
                        if local_epochs_key_str == '1-local-epoch':
                            rounds_1_epoch.append(round_num)
                            log_miss_empty_1_epoch.append(log_miss_empty)
                            log_miss_occupied_1_epoch.append(log_miss_occupied)
                        elif local_epochs_key_str == '2-local-epochs':
                            rounds_2_epoch.append(round_num)
                            log_miss_empty_2_epoch.append(log_miss_empty)
                            log_miss_occupied_2_epoch.append(log_miss_occupied)

                # Plot log-average miss rate for the 'Space-empty' class
                if rounds_1_epoch:
                    axs[row][col].plot(rounds_1_epoch, log_miss_empty_1_epoch, 'r-', label='Empty-1-local-epoch')  # Solid red line
                if rounds_2_epoch:
                    axs[row][col].plot(rounds_2_epoch, log_miss_empty_2_epoch, 'r--', label='Empty-2-local-epochs')  # Dashed red line

                # Plot log-average miss rate for the 'Space-occupied' class
                if rounds_1_epoch:
                    axs[row][col].plot(rounds_1_epoch, log_miss_occupied_1_epoch, 'g-', label='Occupied-1-local-epoch')  # Solid green line
                if rounds_2_epoch:
                    axs[row][col].plot(rounds_2_epoch, log_miss_occupied_2_epoch, 'g--', label='Occupied-2-local-epochs')  # Dashed green line

                # Labeling
                if row == 0:
                    axs[row][col].set_title(f'Client {col + 1}', fontsize=9)
                if col == 0:
                    axs[row][col].set_ylabel(f"{method} Log\n Average Miss Rate\nfor {network_name}", fontsize=9)
                if row == total_rows - 1:
                    axs[row][col].set_xlabel('Round', fontsize=9)
                
                # Set y-axis label and limits
                axs[row][col].set_ylim([-0.1, 1])  # Adjust the limit based on the log range
                
                # Enable grid and set legend
                axs[row][col].grid(True, linestyle='--', linewidth=0.5)
                axs[row][col].legend(loc='upper right') # Uppoer right corner
            
            # Move to the next row for each method
            row += 1

    plt.tight_layout()

    # Save the plot to a PDF file
    miss_rate_img_path = os.path.join(output_dir, 'log_average_miss_rate_per_round.pdf')
    fig.savefig(miss_rate_img_path, bbox_inches='tight')
    print(f"Log Average Miss Rate per round plot saved to {miss_rate_img_path}")
    plt.show()

def visualize_method_per_row(structure_dict):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(dir_path, 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    import pickle
    data_dict = {}
    for network_name in structure_dict:
        data_dict[network_name] = {}
        network_structure_dict = structure_dict[network_name]
        for method in network_structure_dict:
            data_dict[network_name][method] = {}
            for local_epochs_key_str in network_structure_dict[method].keys():
                data_dict[network_name][method][local_epochs_key_str] = {}
                for client_name in ['site-1', 'site-2', 'site-3', 'site-4']:
                    one_epoch_pkl_path = os.path.join(EXPERIMENTS_ROOT_DIR, network_structure_dict[method][local_epochs_key_str], client_name, 'overall_trackers.pkl')
                    with open(one_epoch_pkl_path, "rb") as f:
                        data_dict[network_name][method][local_epochs_key_str][client_name] = pickle.load(f)
    print("Finished reading the pickle files.")
    # calculate_f1_score_iou_50(data_dict, structure_dict)
    # visualize_final_round_f1_score(data_dict, structure_dict, output_dir)
    visualize_final_round_precision_recall(data_dict, structure_dict, output_dir)
    visualize_log_average_miss_rate(data_dict, structure_dict, output_dir)
    method = 'FedAvg'
    client_name = 'site-1'
    # Visualize_method_per_row the losses
    #Create a figure and 4x3 subplots
    fig, axs = plt.subplots(4, 4)
    # Set the size of the figure to the width of a letter size paper
    fig.set_size_inches(12.1, 9)

    y_limits_per_row = []
    for k, network_name in enumerate(structure_dict):
        network_structure_dict = structure_dict[network_name]
        for i, method in enumerate(network_structure_dict):
            for j, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
                max_loss = 0
                for local_epochs_key_str in network_structure_dict[method].keys():
                    max_loss = max(max_loss, max(data_dict[network_name][method][local_epochs_key_str][client_name]['train_loss']))
                y_limits_per_row.append(max_loss)

    # fig.suptitle('Client 1 - FedAvg')
    i = 0
    for network_name in structure_dict:
        network_structure_dict = structure_dict[network_name]
        for method in network_structure_dict:
            for j, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
                for local_epochs_key_str in network_structure_dict[method].keys():
                    num_epochs = len(data_dict[network_name][method][local_epochs_key_str][client_name]['train_loss'])
                    epochs_arr = np.arange(num_epochs)
                    axs[i, j].plot(epochs_arr, data_dict[network_name][method][local_epochs_key_str][client_name]['train_loss'], label=f'{local_epochs_key_str}')
                if i == 0:
                    axs[i, j].set(title=f'Client {j+1}')
                if j == 0:
                    axs[i, j].set(ylabel=f"{method} Loss\nfor {network_name}")
                else:
                    # Remove the y-axis numbers from all subplots except the first one
                    axs[i, j].yaxis.set_tick_params(labelleft=False)

                if i == 3:
                    axs[i, j].set(xlabel="Epoch")
            
                # Set y-axis limits
                axs[i, j].set_ylim([0, y_limits_per_row[i]]) 

                # Display legends in the last column
                if j == 3: # assuming that we have only 4 clients
                    axs[i, j].legend()

                axs[i, j].grid(True, linestyle='--', linewidth=0.5)
            i += 1

    # fig, axs = plt.subplots(4, 3)
    # ax.plot(epochs_arr, data_dict[method]['1-local-epoch'][client_name]['train_loss'], label='1 local epoch')
    # ax.plot(epochs_arr, data_dict[method]['2-local-epochs'][client_name]['train_loss'], label='2 local epochs')
    # ax.set(xlabel="Epoch", ylabel="Loss", title="Client 1 - FedAvg")
    # ax.grid()
    plt.show()

    # Save the plot to a PDF file
    losses_img_path = os.path.join(output_dir, 'fl_losses.pdf')
    fig.savefig(losses_img_path, bbox_inches='tight')
    print(f"Losses plot saved to {losses_img_path}")

    # Draw mAP vs Rounds
    fig, axs = plt.subplots(4, 4)
    # Set the size of the figure to the width of a letter size paper
    fig.set_size_inches(12.1, 8.5)

    y_limits_per_row = []
    for network_name in structure_dict:
        network_structure_dict = structure_dict[network_name]
        for i, method in enumerate(network_structure_dict):
            for j, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
                max_loss = 0
                for local_epochs_key_str in network_structure_dict[method].keys():
                    max_loss = max(max_loss, max(data_dict[network_name][method][local_epochs_key_str][client_name]['train_loss']))
                y_limits_per_row.append(max_loss)

    # fig.suptitle('Client 1 - FedAvg')
    i = 0
    for network_name in structure_dict:
        network_structure_dict = structure_dict[network_name]
        for method in network_structure_dict:
            for j, client_name in enumerate(['site-1', 'site-2', 'site-3', 'site-4']):
                for local_epochs_key_str in network_structure_dict[method].keys():
                    mAP = [x['mAP'] for x in data_dict[network_name][method][local_epochs_key_str][client_name]['val_acc']]
                    # If the method is FedAvg when we have 2 local epochs, take only the even rounds. This is a hack because this was the only experiment were I was evaluating every 1 local epochs instead of every 2 local epochs.
                    if method == 'FedAvg' and local_epochs_key_str == '2-local-epochs':
                        mAP = mAP[1::2]
                    rounds_arr = np.arange(len(mAP))
                    axs[i, j].plot(rounds_arr, mAP, label=f'{local_epochs_key_str}')
                if i == 0:
                    axs[i, j].set(title=f'Client {j+1}')
                if j == 0:
                    axs[i, j].set(ylabel=f"{method} mAP\nfor {network_name}")
                else:
                    # Remove the y-axis numbers from all subplots except the first one
                    axs[i, j].yaxis.set_tick_params(labelleft=False)

                if i == 3:
                    axs[i, j].set(xlabel="Round")
            
                # Set y-axis limits
                axs[i, j].set_ylim([0, 1.1]) 

                # Display legends in the last column
                if j == 3: # assuming that we have only 4 clients
                    axs[i, j].legend(loc='lower right')

                axs[i, j].grid(True, linestyle='--', linewidth=0.5)

            i += 1

    # fig, axs = plt.subplots(4, 3)
    # ax.plot(epochs_arr, data_dict[method]['1-local-epoch'][client_name]['train_loss'], label='1 local epoch')
    # ax.plot(epochs_arr, data_dict[method]['2-local-epochs'][client_name]['train_loss'], label='2 local epochs')
    # ax.set(xlabel="Epoch", ylabel="Loss", title="Client 1 - FedAvg")
    # ax.grid()
    plt.show()

    # Save the plot to a PDF file
    mAP_img_path = os.path.join(output_dir, 'fl_mAP.pdf')
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

            axs[i, j].grid(True, linestyle='--', linewidth=0.5)
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
                # If the method is FedAvg when we have 2 local epochs, take only the even rounds. This is a hack because this was the only experiment were I was evaluating every 1 local epochs instead of every 2 local epochs.
                if method == 'FedAvg' and local_epochs_key_str == '2-local-epochs':
                    mAP = mAP[1::2]
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
                axs[i, j].set_xlabel("Round", fontsize='9')
           
            # Set y-axis limits
            axs[i, j].set_ylim([0.25, 1.1]) 

            axs[i, j].grid(True, linestyle='--', linewidth=0.5)
    # Create a single legend for all subplots and change the font size
    fig.legend(handles, labels, loc='upper left', fontsize='9')
    plt.show()

    # Save the plot to a PDF file
    mAP_img_path = os.path.join(output_dir, 'fl_mAP.pdf')
    fig.savefig(mAP_img_path, bbox_inches='tight')
    print(f"mAP plot saved to {mAP_img_path}")

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

def visualize_local_clients_predictions(predictions_dict):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(dir_path, 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Extract images base directory and sites from predictions_dict
    sites = predictions_dict.get('sites', {})

    # Define number of rows and columns
    num_rows = len(sites)  # 5 rows for each site including 'federated'
    num_cols = len(next(iter(sites.values())))  # 4 columns for each test image

    # Create a figure with a 5x4 grid of subplots with smaller spacing
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 10))

    column_names = predictions_dict.get('column-names', [])

    # Iterate over each site and corresponding images
    for i, (site_name, image_paths) in enumerate(sites.items()):
        for j, (image_path, column_name) in enumerate(zip(image_paths, column_names)):
            # Load the image
            img = mpimg.imread(image_path)

            # Display the image on the subplot
            axs[i, j].imshow(img)
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['bottom'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            axs[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            axs[i, j].set_aspect('auto')  # Ensure all subplots are the same aspect ratio

            # Set the title for the first row only
            if i == 0:
                axs[i, j].set_title(f'{column_name}', fontsize=12)

            # Set the y-axis label for the first column only with site names
            if j == 0:
                axs[i, j].set_ylabel(site_name, fontsize=12, rotation=90, labelpad=15, ha='center', va='center')
                axs[i, j].tick_params(left=False, bottom=False, labelbottom=False)  # Turn off ticks

            # Highlight diagonal images with a rectangle
            if i == j or i == 4:
                rect = patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='blue', facecolor='none',
                                         transform=axs[i, j].transAxes, clip_on=False)
                axs[i, j].add_patch(rect)

    # Add a main title
    # plt.suptitle('Predictions of Test Images for Each Client', fontsize=16)
    
    # Adjust spacing
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.93, bottom=0.05, left=0.08, right=0.98)

    # Show the plot
    plt.show()

    # Save the plot to a PDF file
    img_path = os.path.join(output_dir, 'visualize_image_predections.pdf')
    fig.savefig(img_path, bbox_inches='tight')
    print(f"mAP plot saved to {img_path}")
            


EXPERIMENTS_ROOT_DIR = "/home/bakr/NVFlare/examples/hello-world/parking-federated-training/exprs"
structure_dict = {
    'Resnet50': {
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
    },
    'Sddnet': {
        'SCAFFOLD': {
        '1-local-epoch': 'expr-15',
        # '2-local-epochs': 'expr-15' # TODO: AB: Unitl I finish experiment 16, I use experiment 15 for both 1 and 2 local epochs
        }
    }
}

# visualize_method_per_row(structure_dict)

predictions_dict = {
    'images_base_dir': '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images',
    'sites': {
        'Client 1 Model': ['/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/PUCPR/PUCPR/60.jpg',
                    '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/PUCPR/UFPR04/18.jpg',
                      '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/PUCPR/UFPR05/2.jpg',
                      '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/PUCPR/CNR-EXT/16.jpg'],
        'Client 2 Model': ['/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/UFPR04/PUCPR/60.jpg',
                    '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/UFPR04/UFPR04/18.jpg',
                      '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/UFPR04/UFPR05/2.jpg',
                        '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/UFPR04/CNR-EXT/16.jpg'],
        'Client 3 Model': ['/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/UFPR05/PUCPR/60.jpg', '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/UFPR05/UFPR04/18.jpg',
                    '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/UFPR05/UFPR05/2.jpg',
                      '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/UFPR05/CNR-EXT/16.jpg'],
        'Client 4 Model': ['/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/CNR-EXT/PUCPR/60.jpg',
                    '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/CNR-EXT/UFPR04/18.jpg',
                      '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/CNR-EXT/UFPR05/2.jpg',
                        '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/CNR-EXT/CNR-EXT/16.jpg'],
        'Federated Model': ['/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/PUCPR/PUCPR/60.jpg',
                       '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/UFPR04/UFPR04/18.jpg',
                         '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/UFPR05/UFPR05/2.jpg',
                           '/home/bakr/NVFlare/examples/hello-world/parking-federated-training/saved_debug_images/CNR-EXT/CNR-EXT/16.jpg']
    },
    'column-names': ['Client 1 Test Image (PUCPR)', 'Client 2 Test Image (UFPR04)', 'Client 3 Test Image (UFPR05)', 'Client 4 Test Image (CNR-EXT)']
}
# visualize_local_clients_predictions(predictions_dict) # TODO: Implement this function
visualize_clients_per_row(structure_dict)
