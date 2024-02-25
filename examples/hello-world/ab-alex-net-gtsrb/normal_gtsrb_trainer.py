import os.path

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torch.utils.data import random_split
import torchvision
import time
import pickle
import matplotlib.pyplot as plt

import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, 'jobs/ab-alex-net-gtsrb/app/custom/')))
print(sys.path)
# import alex_net_network
from alex_net_network import AlexnetTS
from gtsrb_TestDataLoader import GTSRB_TestDataLoader

data_path="~/data/gtsrb/GTSRB"
users_split = 2
batch_size = 4

class GTSRB:
    def __init__(
        self,
        data_path="~/data/gtsrb/GTSRB",
        lr=0.01,
        epochs=2,
        num_classes = 43,
        batch_size = 128,
        train_val_split = 0.8,
        random_seed = 42,
        load_model_from_disk = False,
        model_load_path = None
    ):
        """GTSRB Trainer handles train and submit_model tasks. During train_task, it trains a
        simple network on GTSRB dataset. For submit_model task, it sends the locally trained model
        (if present) to the server.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            train_task_name (str, optional): Task name for train task. Defaults to "train".
            submit_model_task_name (str, optional): Task name for submit model. Defaults to "submit_model".
            train_val_split: This is the percentage of the data that will be used for the training. The rest will be used for the validation.
            random_seed: This is the seed that will be used to split the data into training and validation.
            load_model_from_disk: This is a boolean value that indicates whether the model should be loaded from the disk or not.
            model_load_path: This is the path of the model that should be loaded from the disk. Only used if load_model_from_disk is True.
        """
        super().__init__()

        # AB: Parameters
        
        users_split = 2 # AB: This is the number of clients that will be used for the training. It is set to 2, so that the data will be split between two clients.
        torch.manual_seed(random_seed) # AB: This is to make sure that the training and the validation data are split in the same way for all the clients.
        
        self._lr = lr
        self._epochs = epochs
        self.model_save_path = os.path.abspath(os.path.join(dir_path, "model.pth"))
        self.train_trackers_filename = os.path.abspath(os.path.join(dir_path, 'normal_train_trackers.p'))

        print(f"Number of classes: {num_classes}, Learning rate: {lr}, Number of epochs: {epochs}, Batch size: {batch_size}, Train validation split: {train_val_split}, Users split: {users_split}, data path: {data_path}")

        # Training setup
        self.model = AlexnetTS(num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)

        if load_model_from_disk:
            model_path = model_load_path if model_load_path else os.path.abspath(os.path.join(dir_path, "model.pth"))
            self.load_model_from_disk(model_path)

        # Create GTSRB dataset for training.
        transforms = Compose(
            [
                Resize([112, 112]),
                ToTensor()
            ]
        )
        train_data_path = os.path.join(data_path, "Training")
        self._dataset = torchvision.datasets.ImageFolder(root = train_data_path, transform = transforms)

        n_train_examples = int(len(self._dataset) * train_val_split)
        n_val_examples = len(self._dataset) - n_train_examples

        self._train_dataset, self._val_dataset = random_split(self._dataset, [n_train_examples, n_val_examples])

        self._train_loader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True)
        self._validate_loader = DataLoader(self._val_dataset, batch_size=batch_size, shuffle=False)
        self._n_iterations = len(self._train_loader)
        print(f"Number of iterations: {self._n_iterations}")
        print(f"Number of samples from the whole data: {len(self._train_dataset.indices)} = Number of iterations ({self._n_iterations}) * batch size ({batch_size})")

        print(f"Gtsrb43Trainer initialized: This is the path of the data: {data_path}")

        test_data_path = os.path.join(data_path, "Final_Test")
        test_dataset = GTSRB_TestDataLoader(data_path, transform = transforms)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def load_model_from_disk(self, model_path):
        if os.path.exists(model_path):
            print(f"Model exists on the disk with the name: {model_path}")
            self.model.load_state_dict(torch.load(model_path))
            print("Model loaded from disk")
        else:
            print(f"Model does not exist on the disk with the name: {model_path}")

    
    # Function to perform training of the model
    def local_train(self, validate=True, save_graphs_after_each_epoch=False):
        trackers = {"train_loss": [], "val_acc": []}
        # Basic training
        self.model.train()
        for epoch in range(self._epochs):
            running_loss = 0.0
            train_start_time = time.monotonic()
            for i, batch in enumerate(self._train_loader):
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()

                predictions = self.model(images)
                cost = self.loss(predictions, labels)
                cost.backward()
                self.optimizer.step()

                running_loss += cost.cpu().detach().numpy() / images.size()[0]
                # if i % 3000 == 0:
                #     print(f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, " f"Loss: {(running_loss/3000)}")
                #     running_loss = 0.0
                    # Write this model to the disk
            epoch_loss = running_loss / len(self._train_loader)
            trackers['train_loss'].append(epoch_loss)
            print(f"Epoch: {epoch}/{self._epochs} ended, Loss: {trackers['train_loss'][epoch]}, Time: {(time.monotonic() - train_start_time):.2f} seconds")
            torch.save(self.model.state_dict(), self.model_save_path) # AB: Save the model to the disk after each epoch
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), os.path.abspath(os.path.join(dir_path, f"model_{epoch}.pth")))
            if validate:
                val_start_time = time.monotonic()
                trackers['val_acc'].append(self.validate(self._validate_loader))
                val_time = time.monotonic() - val_start_time
                print(f"Validation accuracy (correct / total validation images): {(trackers['val_acc'][epoch] * 100):.2f}%, Time: {val_time:.2f} seconds")
            pickle.dump(trackers, open(self.train_trackers_filename, 'wb')) # AB: Save the training trackers to the disk after each epoch
            if save_graphs_after_each_epoch:
                self.display_train_trackers()

    def validate(self, loader):
        """
        Validate the model on the given loader (validate or test).
        """
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)

                _, pred_label = torch.max(output, 1)

                correct += (pred_label == labels).sum().item()
                total += images.size()[0]
            print(f"Correct: {correct}, not Correct: {total - correct}, Total: {total}")
            metric = correct / float(total)

        return metric
    
    def display_train_trackers(self):
        trackers = pickle.load(open(self.train_trackers_filename, 'rb'))
        # print(trackers)
        # Display train loss graph.
        plt.figure() # New graph
        plt.plot(trackers['train_loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Train loss')
        plt.title('Train loss vs Epochs')
        # Save the graph to the disk
        plt.savefig(os.path.abspath(os.path.join(dir_path, 'normal_train_loss.png')))

        # Display validation accuracy graph.
        plt.figure() # New graph
        plt.plot(trackers['val_acc'])
        plt.xlabel('Epochs')
        plt.ylabel('Validation accuracy')
        plt.title('Validation accuracy vs Epochs')
        # Save the graph to the disk
        plt.savefig(os.path.abspath(os.path.join(dir_path, 'normal_val_acc.png')))

if __name__ == "__main__":
    # Resolve ~ to the home directory
    data_path = os.path.expanduser("~/data/gtsrb/GTSRB")
    print(data_path)
    start_time = time.monotonic()
    gtsrb = GTSRB( data_path=data_path,
                    lr=0.01,
                    epochs=40,
                    batch_size = 128,
                    train_val_split = 0.8,
                    load_model_from_disk = False, # If True, the model will be loaded from the disk
                    model_load_path = os.path.abspath(os.path.join(dir_path, "model.pth"))) # The path of the model that should be loaded from the disk if load_model_from_disk is True

    gtsrb.local_train(validate=True, save_graphs_after_each_epoch=True) # AB: Training the model
    gtsrb.display_train_trackers() # AB: Display the training trackers

    validation_on_test_accuracy = gtsrb.validate(gtsrb.test_loader) * 100 # AB: Final validation on the test data
    print(f"Validation accuracy on test data: {validation_on_test_accuracy:.2f}%")
    print(f"Total running time: {(time.monotonic() - start_time):.2f} seconds")
