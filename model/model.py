import torch
import torch.nn as nn

### Model : RRCNN-C ###

class ResBlock(nn.Module):
  def __init__(self, num_ft = 64, kernel_size = 3, stride = 1, padding = 1):
    super(ResBlock, self).__init__()
    m = []
    for i in range(2):
      m.append(nn.Conv1d(num_ft, num_ft, kernel_size, stride, padding))
      m.append(nn.BatchNorm1d(num_ft))
      m.append(nn.ReLU())
    self.body = nn.Sequential(*m)

  def forward(self, x):
    res = self.body(x)
    res += x
    return res


class RRCNN_C(nn.Module):
  def __init__(self, num_channels,  num_classes, num_res_ft = 64, num_res = 2):
    super(RRCNN_C, self).__init__()
    self.conv = nn.Conv1d(num_channels, num_res_ft, kernel_size = 3, stride = 1, padding = 1)
    self.res = ResBlock()
    mat = []
    for _ in range(num_res):
      mat.append(ResBlock(num_ft = num_res_ft))
      mat.append(nn.RReLU())
    self.res_body_1 = nn.Sequential(*mat) 

    mat2 = []
    for _ in range(num_res):
      mat2.append(ResBlock(num_ft = num_res_ft))
      mat2.append(nn.RReLU())
    self.res_body_2 = nn.Sequential(*mat2) 

    mat3 = []
    for _ in range(num_res):
      mat3.append(ResBlock(num_ft = num_res_ft))
      mat3.append(nn.RReLU())
    self.res_body_3 = nn.Sequential(*mat3) 

    self.avg = nn.AdaptiveAvgPool1d(1)
    self.maxpool = nn.MaxPool1d(1)

    self.fc = nn.Linear(num_res_ft, num_classes)
    self.clf = nn.Softmax()

  def forward(self, x):
    x_in = self.conv(x)
    x = self.res_body_1(x_in)
    x = x + x_in
    x1 = x
    x = self.res_body_2(x)
    x = x+x1
    x2 = x
    x = self.res_body_3(x)
    x = x + x2
    x = self.avg(x)
    x = torch.flatten(x)
    x = self.fc(x)
    x = self.clf(x)
    return x