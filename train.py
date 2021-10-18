import os
import numpy as np
import argparse
import json
import torch
import torch.nn as nn

from model.models import model_version
from dataset.dataset import dataset
from utils.optimizer import optimizer_function
from utils.loss import loss_function


parser = argparse.ArgumentParser(description="Training RRCNN")
# Dataset paths
parser.add_argument('-d_train', '--train_data_path', type=str, default='./data/Train data/train_data.pkl',
                    help='The path of test data')
parser.add_argument('-l_train', '--train_label_path', type=str, default='./data/Train data/train_label.pkl',
                    help='The path of test labels')
parser.add_argument('-d_val', '--test_data_path', type=str, default='./data/Test data/test_data.pkl',
                    help='The path of test data')
parser.add_argument('-l_val', '--test_label_path', type=str, default='./data/Test data/test_label.pkl',
                    help='The path of test labels')
parser.add_argument('-c', '--config', type = str, default= './config/config.json', help='Path to the config file')

# Training parameters
parser.add_argument('-m', '--model', type = str, default = 'RRCNN_C', help = 'Choose model version')
parser.add_argument('-lr', '--lr', type = float, default = 0.0001, help = 'Learning rate')
parser.add_argument('-n', '--num_classes', type = int, default = 2, help = 'The number of classes')
parser.add_argument('-mm', '--momentum', type = float, default = 0.58, help = 'The momentum of the optimizer')
parser.add_argument('-opt', '--optimizer', type = str, default='Adam', help = 'Choose optimizer')
parser.add_argument('-ne', '--epoch', type = int, default = 300, help = 'No. of epochs')
parser.add_argument('-lss', '--loss', type = str, default = 'Cross entropy loss', help = 'The loss function')
parser.add_argument('-mp', '--model_path', type = str, default = './model weights', 
                    help='Path where the model weights are saved')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataloader
train_data_path = args.train_data_path
train_label_path = args.train_label_path
train_data, train_label = dataset(train_data_path, train_label_path)

test_data_path = args.test_data_path
test_label_path = args.test_label_path
test_data, test_label = dataset(test_data_path, test_label_path)

config = json.load(args.config)
models_params = config[args.model]

num_epoch = args.epoch
lr = args.lr
momentum = args.momentum
num_channels = models_params.num_channels
num_residual_features = models_params.num_residual_features
num_resedual_blocks = models_params.num_resedual_blocks
num_classes = args.num_classes


model = model_version(num_channels = num_channels,  num_classes = num_classes, 
                num_res_ft = num_residual_features, num_res = num_resedual_blocks, model = args.model)


model = model.to(device = device)
params = model.parameters()
optm = args.optimizer
optimizer = optimizer_function(params, lr, momentum, optm)
criterion = loss_function(args.loss)


model_name = args.model + '_train'
snapshot_path = args.model_path

load_model=snapshot_path+'/model_'+model_name+'.pth'
loaded_flag=False
if os.path.exists(load_model):
    checkpoint=torch.load(load_model)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print("model loaded successfully")
    print('starting training after epoch: ',checkpoint['epoch'])
    loaded_flag=True


max_acc = 0.0

for epoch in range(num_epoch):
  train_loss = 0.0
  correct = total = 0
  for i in range(len(train_data)):
    optimizer.zero_grad()
    data_point, label = torch.tensor(train_data[i]), torch.tensor(np.array([train_label[i]]))
    data_point, label = data_point.to(device=device), label.to(device=device)
    data_point = data_point.reshape(1,num_channels,-1)
    output = model(data_point.float())
    loss = criterion(output.reshape(1,-1), label)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    _, predicted = torch.max(output.reshape(1,-1).data, 1)
    total += label.size(0)
    correct += (predicted == label).sum().item()

  print('Training Epoch: ', epoch)
  print('training loss: ', train_loss)
  print('Accuracy: ', 100*correct/total)

  with torch.no_grad():
    val_loss = 0.0
    total = correct = 0
    for j in range(len(test_data)):
      val_data, val_label = torch.tensor(test_data[j]), torch.tensor(np.array([test_label[j]]))
      val_data, val_label = val_data.to(device=device), val_label.to(device=device)
      val_data = val_data.reshape(1,num_channels,-1)
      out_val = model(val_data.float())
      loss = criterion(out_val.reshape(1,-1), val_label)
      val_loss += loss.item()
      _, predicted_val = torch.max(out_val.reshape(1,-1).data, 1)
      total += val_label.size(0)
      correct += (predicted_val == val_label).sum().item()
  print('Validation Loss: ', val_loss)
  print('Validation Accuracy: ', 100*correct/total)
  val_acc = 100*correct/total

  if val_acc>max_acc:
    state={
        "epoch":i if not loaded_flag else i+checkpoint['epoch'],
        "model_state":model.cpu().state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "Accuracy":val_acc
        #"loss":min_loss,
        #"train_graph":train_loss_gph,
        #"val_graph":val_loss_gph,
    }

    max_acc=val_acc
    torch.save(state,os.path.join(snapshot_path,"model_"+model_name+'.pth'))
    model.cuda()
  print('maximum validation accuracy : ', max_acc)


