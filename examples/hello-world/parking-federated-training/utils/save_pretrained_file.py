import torch

import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '../jobs/parking-federated-training/app/custom/')))
print(sys.path)
import Resnet as resnet

def save_pretrained_file(num_classes, path):
    model = resnet.ResnetFasterRCNN.get_pretrained_model(3)
    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    num_classes = 3
    output_path = os.path.join(dir_path, 'resnet_pretrained_model.pth')
    save_pretrained_file(3, output_path)
    print(f"Pretrained model saved to {output_path}")
