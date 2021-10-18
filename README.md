# Motivation Detection using RRCNN Model
This code refers to the paper entitled [Motivation detection using EEG signal analysis by residual-in-residual convolutional neural network](https://www.sciencedirect.com/science/article/pii/S0957417421009544 "link to the paper"), where the motivation level of different test subjects is detected by analyzing Electroencephalography Signals. Specifically, the proposed RRCNN model detects whether a person is enough motivated to play a reward-based game (designed in the NTU lab) or not. Here the proposed model is residual-in-residual 1D CNN or RRCNN, where dual residual operation is used for better feature learning from multi-channeled EEG data. 

https://codeocean.com/capsule/0422935/tree

In the CodeOcean platform, the test data, trained weights and the complete test code is given. Just by clicking on the Reproducible Run button, the test results can be achieved. It is to be noted that, by performing several test runs, the most occuring best test accuracy obtained is 89%.  

## Dependencies
- Preferably [`python 3.6.8`](https://www.python.org/downloads/release/python-368/) or above.
- pytorch : `pip3 install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio===0.9.1 -f https://download.pytorch.org/whl/torch_stable.html`
- numpy : `pip3 install numpy==1.20.3`
- pandas : `pip3 install pandas==1.2.3`

## Data
The test data and labels can be found [here](https://codeocean.com/capsule/0422935/tree). The entire dataset will be published soon after some mandetory preprocessing of it.

## Execution
- ### Test : 
  - #### Dataset : 
  - #### Code execution : 
         python3 test.py -m RRCNN_C -n 2
