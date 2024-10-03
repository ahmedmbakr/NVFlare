import torch

import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '../jobs/parking-federated-training/app/custom/')))
print(sys.path)
import Resnet as resnet
import SSDnet as ssdnet

def save_pretrained_file(model_name, num_classes, path):
    if model_name == 'ssdnet':
        model = ssdnet.SSDVGG16.get_pretrained_model(num_classes)
    elif model_name == 'resnet':
        model = resnet.ResnetFasterRCNN.get_pretrained_model(num_classes)
    else:
        raise ValueError(f"Model name {model_name} is not supported. The supported model names are 'ssdnet' and 'resnet'.")
    torch.save(model.state_dict(), path)
    

if __name__ == '__main__':
    num_classes = 3
    model_name = 'ssdnet' # The options are 'ssdnet' and 'resnet'
    output_path = os.path.join(dir_path, f'{model_name}_pretrained_model.pth')
    save_pretrained_file(model_name, num_classes, output_path)
    print(f"Pretrained model saved to {output_path}")
