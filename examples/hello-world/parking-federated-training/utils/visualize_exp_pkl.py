# The purpose of this script is to visualize the experiment results stored in a pickle file.

def visualize_pkl_file(pickle_file_path, save_folder_path=None):
    """
    Visualize the data stored in a pickle file.
    Args:
        pickle_file_path: The path to the pickle file.
    """
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np

    with open(pickle_file_path, "rb") as f:
        data = pickle.load(f)

    num_epochs = len(data['train_loss'])
    epochs_arr = np.arange(num_epochs)

    # Visualize the losses
    fig, ax = plt.subplots()
    ax.plot(epochs_arr, data['train_loss'])
    ax.set(xlabel="Epoch", ylabel="Loss", title="Loss vs Epoch")
    ax.grid()
    plt.show(block=True)
    # save the figure
    if save_folder_path is not None:
        fig.savefig(save_folder_path + "/loss_vs_epoch.png")
    
    # Visualize the APs
    space_empty_ap = [x['ap']['Space-empty'] for x in data['val_acc']]
    space_occupied_ap = [x['ap']['Space-occupied'] for x in data['val_acc']]
    mAPs = [x['mAP'] for x in data['val_acc']]
    max_mAP = max(mAPs)
    max_mAP_index = mAPs.index(max_mAP) # Get index of the maximum mAP
    fig, ax = plt.subplots()
    rounds_arr = np.arange(len(mAPs))
    ax.plot(rounds_arr, space_empty_ap, label="Space-empty AP")
    ax.plot(rounds_arr, space_occupied_ap, label="Space-occupied AP")
    ax.plot(rounds_arr, mAPs, label="mAP")
    ax.plot(max_mAP_index, max_mAP, 'xg') # Draw a mark on the maximum point
    # Put text on top of this point with the maximum mAP and make the text color green
    ax.annotate(f"({max_mAP_index},{max_mAP:0.4f})", (max_mAP_index, max_mAP), textcoords="offset points", xytext=(0,5), ha='center', color='g')
    # Draw a vertical line from the maximum point to the x-axis
    ax.axvline(x=max_mAP_index, color='g', linestyle='--')
    ax.legend() #show legend
    # Draw a mark on the maximum point
    ax.set(xlabel="Epoch", ylabel="mAP", title="mAP vs Epoch")
    # Set the y axis limits to be between 0 and 1
    # ax.set_ylim([0, 1])
    ax.grid()
    plt.show(block=True)
    # save the figure
    if save_folder_path is not None:
        fig.savefig(save_folder_path + "/mAP_vs_epoch.png")

    # Visualize log average miss rate
    log_avg_miss_rate_space_empty = [x['log_avg_miss_rate']['Space-empty'] for x in data['val_acc']]
    log_avg_miss_rate_space_occupied = [x['log_avg_miss_rate']['Space-occupied'] for x in data['val_acc']]
    fig, ax = plt.subplots()
    ax.plot(rounds_arr, log_avg_miss_rate_space_empty, label="Space-empty")
    ax.plot(rounds_arr, log_avg_miss_rate_space_occupied, label="Space-occupied")
    ax.plot(max_mAP_index, max_mAP, 'xg') # Draw a mark on the maximum point
    # Draw a vertical line from the maximum point to the x-axis
    ax.axvline(x=max_mAP_index, color='g', linestyle='--', label="Max mAP")
    # Put text on top of this point with the maximum mAP and make the text color green
    ax.annotate(f"({max_mAP_index},{max_mAP:0.4f})", (max_mAP_index, max_mAP), textcoords="offset points", xytext=(35,-3), ha='center', color='g')
    ax.legend() #show legend
    # Draw a mark on the maximum point
    ax.set(xlabel="Epoch", ylabel="Log Avg. Miss Rate", title="Log Average Miss Rate vs Epoch")
    ax.grid()
    plt.show(block=True)
    # save the figure
    if save_folder_path is not None:
        fig.savefig(save_folder_path + "/log_avg_miss_rate_vs_epoch.png")



if __name__ == "__main__":
    # Read a pickle file
    pickle_file_path = "/home/bakr/NVFlare/examples/hello-world/parking-federated-training/outputs/overall_trackers.pkl"
    visualize_pkl_file(pickle_file_path)
