import os
import numpy as np
import argparse
import json
import torch
import torch.nn as nn

from model.models import model_version
from dataset.dataset import test_dataset

parser = argparse.ArgumentParser(description="Testing RRCNN")
parser.add_argument('-d', '--data_path', type=str, default='./data/Test data/test_data.pkl',
                    help='The path of test data')
parser.add_argument('-l', '--label_path', type=str, default='./data/Test data/test_label.pkl',
                    help='The path of test labels')
parser.add_argument('-c', '--config', type = str, default= './config/config.json', help='Path to the config file')
parser.add_argument('-m', '--model', type = str, default = 'RRCNN_C', help = 'Choose model version')
parser.add_argument('-n', '--num_classes', type = int, default = 2, help = 'The number of classes')
parser.add_argument('-mp', '--model_path', type = str, default = './model weights', 
                    help='Path where the model weights are saved')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = json.load(args.config)
models_params = config[args.model]

test_data_path = args.data_path
test_label_path = args.label_path

test_data, test_label = test_dataset(test_data_path, test_label_path)


num_channels = models_params.num_channels
num_residual_features = models_params.num_residual_features
num_resedual_blocks = models_params.num_resedual_blocks
num_classes = args.num_classes

model = model_version(num_channels = num_channels,  num_classes = num_classes, 
                num_res_ft = num_residual_features, num_res = num_resedual_blocks, model = args.model)

model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()



model_name = args.model + '_train'
snapshot_path = args.model_path

load_model=snapshot_path+'/model_'+model_name+'.pth'
loaded_flag=False
if os.path.exists(load_model):
    checkpoint=torch.load(load_model) #torch.jit.load()
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print("model loaded successfully")
    print('starting training after epoch: ',checkpoint['epoch'])
    loaded_flag=True


with torch.no_grad():
  val_loss = 0.0
  total = correct = 0
  for j in range(len(test_data)):
    val_data, val_label = torch.tensor(test_data[j]), torch.tensor(np.array([test_label[j]]))
    val_data, val_label = val_data.to(device = device), val_label.to(device = device)
    val_data = val_data.reshape(1,num_channels,-1)
    out_val = model(val_data.float())
    loss = criterion(out_val.reshape(1,-1), val_label)
    val_loss += loss.item()
    _, predicted_val = torch.max(out_val.reshape(1,-1).data, 1)
    total += val_label.size(0)
    correct += (predicted_val == val_label).sum().item()
print('Test Loss: ', val_loss)
print('Test Accuracy: ', 100*correct/total)
val_acc = 100*correct/total
