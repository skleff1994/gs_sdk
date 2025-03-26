import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from calibration.utils import load_csv_as_dict, transfer_weights
from calibration.models import BGRXYDataset, BGRXYMLPNet_
from gs_sdk.gs_reconstruct import BGRXYMLPNet, Reconstructor

calib_dir = "/home/skleff/panda_ws/src/gs_sdk/calibration/test/"

def compare_models():

    # def set_seed(seed=42):
    #     np.random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False  # This ensures deterministic behavior

    # # Set seed at the beginning of your script
    # set_seed(12)


    # Load the train and test split
    data_path = os.path.join(calib_dir, "train_test_split.json")
    with open(data_path, "r") as f:
        data = json.load(f)
        test_reldirs = data["test"]
    
    # Load data 
    test_data = {"all_bgrxys": [], "all_gxyangles": []}
    for experiment_reldir in test_reldirs:
        data_path = os.path.join(calib_dir, experiment_reldir, "data.npz")
        if not os.path.isfile(data_path):
            raise ValueError("Data file %s does not exist" % data_path)
        data = np.load(data_path)
        test_data["all_bgrxys"].append(data["bgrxys"][data["mask"]])
        test_data["all_gxyangles"].append(data["gxyangles"][data["mask"]])
    test_bgrxys = np.concatenate(test_data["all_bgrxys"])
    test_gxyangles = np.concatenate(test_data["all_gxyangles"])

    # Create train and test Dataloader
    test_dataset = BGRXYDataset(test_bgrxys, test_gxyangles)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

    recon_custom = Reconstructor("/home/skleff/panda_ws/src/gs_sdk/calibration/test/model/nnmodel_custom.pth")
    recon_default = Reconstructor("/home/skleff/panda_ws/src/gs_sdk/calibration/test/model/nnmodel_default.pth")
    
    net_custom_mlp = BGRXYMLPNet_().to("cuda")
    net_custom_cnn = recon_custom.gxy_net
    net_default_mlp = BGRXYMLPNet_().to("cuda")
    net_default_cnn = recon_default.gxy_net
    transfer_weights_cnn_to_mlp(net_custom_cnn, net_custom_mlp)
    transfer_weights_cnn_to_mlp(net_default_cnn, net_default_mlp)

    # Initial evaluation
    test_mae_custom = evaluate(net_custom_mlp, test_dataloader, "cuda")
    test_mae_default = evaluate(net_default_mlp, test_dataloader, "cuda")
    print("Test MAE custom NN : %.4f" % (test_mae_custom))
    print("Test MAE default NN : %.4f" % (test_mae_default))



def evaluate(net, dataloader, device):
    """
    Evaluate the network loss on the dataset.

    :param net: nn.Module; the network to evaluate.
    :param dataloader: DataLoader; the dataloader for the dataset.
    :param device: str; the device to evaluate the network.
    """
    net.to("cuda")
    net.eval()  # Set the model to evaluation mode
    losses = []
    for bgrxys, gxyangles in dataloader:
        bgrxys = bgrxys.to(device)
        gxyangles = gxyangles.to(device)
        outputs = net(bgrxys)
        diffs = outputs - gxyangles
        losses.append(np.abs(diffs.cpu().detach().numpy()))
    mae = np.mean(np.concatenate(losses))
    return mae


# Function to transfer weights from the convolutional model (BGRXYMLPNet) to the MLP model (BGRXYMLPNet_)
def transfer_weights_cnn_to_mlp(cnn_model, mlp_model):
    """
    Transfers weights from the convolutional model (BGRXYMLPNet) to the fully connected MLP model (BGRXYMLPNet_).
    """
    with torch.no_grad():
        cnn_weights = list(cnn_model.parameters())
        mlp_weights = list(mlp_model.parameters())

        # Transfer weights from conv1 to fc1
        mlp_weights[0].data.copy_(cnn_weights[0].data.view_as(mlp_weights[0].data))  # Conv1 to FC1
        mlp_weights[1].data.copy_(cnn_weights[1].data.view_as(mlp_weights[1].data))  # BN1
        mlp_weights[2].data.copy_(cnn_weights[2].data.view_as(mlp_weights[2].data))  # Conv2 to FC2
        mlp_weights[3].data.copy_(cnn_weights[3].data.view_as(mlp_weights[3].data))  # BN2
        mlp_weights[4].data.copy_(cnn_weights[4].data.view_as(mlp_weights[4].data))  # Conv3 to FC3
        mlp_weights[5].data.copy_(cnn_weights[5].data.view_as(mlp_weights[5].data))  # BN3
        mlp_weights[6].data.copy_(cnn_weights[6].data.view_as(mlp_weights[6].data))  # Conv4 to FC4

    return mlp_model

if __name__ == "__main__":
    compare_models()
