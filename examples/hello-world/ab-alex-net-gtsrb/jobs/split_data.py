import os
import shutil
import random

def split_data(data_path, num_clients, output_path):
    """
    Split the training data between multiple clients.
    This function creates a new folder for each client and copies a portion of the training data to each client folder.
    Args:
        data_path: The path to the training data.
        num_clients: The number of clients to split the data between.
        output_path: The path to the output folder.
    """
    client_folder_pattern = output_path + "/site-{}/data"

    # if os.path.exists(output_path):
    #     # Remove the output directory if it exists.
    #     shutil.rmtree(output_path)
    for i in range(1, num_clients + 1):
        client_path = client_folder_pattern.format(i)
        os.makedirs(client_path, exist_ok=True)
    
    for class_dir in os.listdir(data_path):
        class_path = os.path.join(data_path, class_dir)
        if not os.path.isdir(class_path):
            continue
        # print("Class path: ", class_path)
        class_files = os.listdir(class_path)
        random.shuffle(class_files)
        num_files = len(class_files)
        for i in range(num_clients):
            start = i * num_files // num_clients
            end = (i + 1) * num_files // num_clients
            output_folder_path = client_folder_pattern.format(i + 1) + "/" + class_dir
            os.makedirs(output_folder_path, exist_ok=True)
            for j in range(start, end):
                # Copy the file to the client folder.
                source_file = os.path.join(class_path, class_files[j])
                if "ppm" not in source_file: # AB: I added this line to skip the files that are not images
                    continue
                destination_file_path = os.path.join(output_folder_path, class_files[j])
                shutil.copy(source_file, destination_file_path)
                print(f"Copying {source_file} to {destination_file_path}")
    

if __name__ == "__main__":
    # Split the training data between multiple clients.
    num_clients = 2
    data_path = "../data/gtsrb/GTSRB/Training"
    # output_path = "../data/gtsrb/GTSRB/Federated_training"
    output_path = "/tmp/nvflare/poc/example_project/prod_00/"
    random_seed = 42

    random.seed(random_seed)
    split_data(data_path, num_clients, output_path)
    print(f"Data split successfully inside the output folder: {output_path}")
