import os
import pickle
import numpy as np
import pandas

def test_dataset(data_path, label_path):
    with open(data_path, 'rb') as f:
        df_data = pickle.load(f)

    with open(label_path, 'rb') as f:
        df_label = pickle.load(f)

    temp_data_df = df_data[0]
    temp_data = np.array(temp_data_df.values.tolist()) 

    temp_label_df = df_label
    temp_labels = temp_label_df.values.tolist()

    temp_label = []
    for label in temp_labels:
        temp_label.append(label[0])

    return temp_data, temp_label