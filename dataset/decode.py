import wfdb
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
def pickle_load(data_path):
    with open(data_path, 'rb') as f:
        try:
            pickle_data = pickle.load(f)
        except ValueError:
            print("error in ", data_path)
            sys.exit()
    return pickle_data
def pickle_write(python_data, pickle_path):
    with open(pickle_path, 'wb') as f:
        try:
            pickle.dump(python_data, f)
        except ValueError:
            print("error in ", json_path)
            sys.exit()
def print_signal():
    record = wfdb.rdrecord('./mit-bih-arrhythmia-database-1.0.0/108',\
        sampfrom=0, sampto=1000, physical=True) 
        #sampfrom=0, sampto=10000, physical=False, channels=[0, 1])
    print(record.__dict__)
    # annotation = wfdb.rdann('./mit-bih-arrhythmia-database-1.0.0/100', "atr", sampfrom=0, sampto=1000)
    ventricular_signal = record.p_signal
    print('signal shape: ' + str(ventricular_signal.shape)) # 繪製波形
    plt.plot(ventricular_signal[:,0], color="blue")
    plt.plot(ventricular_signal[:,1], color="red")
    plt.title("ventricular signal")
    plt.show()

def ann_statistics(data_dir):
    ann_stat = {}
    for data_name in os.listdir(data_dir):
        if "atr" in data_name:
            print(data_name)
            annotation = wfdb.rdann(data_dir+data_name.split(".atr")[0], "atr")
            ann_list = annotation.symbol[1:]
            for ann in ann_list:
                if ann in ann_stat:
                    ann_stat[ann] += 1
                else:
                    ann_stat[ann] = 1
    print(ann_stat)
def build_dataset(data_dir):
    signal_dict = {}
    ann_dict = {}#{ann: [[data_name, start_index, end_index],...]}
    for raw_data_name in os.listdir(data_dir):
        if "dat" in raw_data_name:
            print(raw_data_name)
            data_name = raw_data_name.split(".dat")[0]
            data_path = data_dir+data_name
            #signal
            record = wfdb.rdrecord(data_path, physical=True)
            ventricular_signal = record.p_signal
            signal_dict[data_name] = ventricular_signal
            #ann
            annotation = wfdb.rdann(data_path, "atr")
            sample_list = annotation.sample
            ann_list = annotation.symbol[1:]#the first is null
            for ann_index, ann in enumerate(ann_list):
                if ann_index==0:
                    continue #the first wave is not complete, so remove it.
                if ann not in ann_dict:
                    ann_dict[ann] = []
                ann_dict[ann].append([data_name, sample_list[ann_index], sample_list[ann_index+1]])
    #separate to training and testing
    train_ann_dict = {}
    val_ann_dict = {}
    train_cut_ratio = 0.9
    for ann_key, ann_list in ann_dict.items():
        list_len = len(ann_list)
        index_list = [i for i in range(list_len)]
        np.random.shuffle(index_list)
        train_cut_index = (int)(list_len*train_cut_ratio)
        train_ann_list = [ann_list[index] for index in index_list[:train_cut_index]]
        train_ann_dict[ann_key] = train_ann_list

        val_ann_list = [ann_list[index] for index in index_list[train_cut_index:]]
        val_ann_dict[ann_key] = val_ann_list

    train_dataset = {
        "signal":signal_dict,
        "annotation":train_ann_dict
    }
    pickle_write(train_dataset, "./train_dataset.pickle")

    val_dataset = {
        "signal":signal_dict,
        "annotation":val_ann_dict
    }
    pickle_write(val_dataset, "./val_dataset.pickle")

    # dataset = {
    #     "signal":signal_dict,
    #     "annotation":ann_dict
    # }
    # pickle_write(dataset, "./dataset.pickle")
def read_dataset(data_path):
    dataset = pickle_load(data_path)
    signal_dict = dataset["signal"]
    ann_dict = dataset["annotation"]
    print(ann_dict.keys())
    #print wave in normal
    print(len(ann_dict["N"]))
    for data_set in ann_dict["N"]:
        plt.clf()
        print(data_set)
        plt.plot(signal_dict[data_set[0]][data_set[1]:data_set[2], 0], color="blue")
        plt.title("ventricular signal")
        plt.show()

# build_dataset('./mit-bih-arrhythmia-database-1.0.0/')
# read_dataset("./train_dataset.pickle")
print_signal()

