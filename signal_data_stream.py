from data import *
import os
import numpy as np
import time
import cv2
import random
import csv
import json
from multiprocessing import Process, Queue
from collections import defaultdict
#load data from pickle, so there is no dataset.py

class SignalDataGenerator():
    def __init__(self, data_name, dataset_path="./dataset/", uniform_class=False, random_shift=False, shuffle=False):
        self.data_name = data_name
        self.dataset_path = dataset_path
        self.random_shift = random_shift
        self.shuffle = shuffle
        self.uniform_class = uniform_class
    def data_read(self):
        dataset = pickle_load(self.dataset_path+self.data_name+".pickle")
        signal_dict = dataset["signal"]
        ann_dict = dataset["annotation"]
        return ann_dict, signal_dict
    def get_data_len(self, label_list):
        label_dict, signal_dict = self.data_read()
        data_len = 0
        for label in label_list:
            data_len += len(label_dict[label])
        return data_len
    def set_signal(self, signal, interval, origin_len, downsample_count):#clip and, if needed, shift
        if self.random_shift:
            random_shift_ratio = 0.1
            signal_len = interval[1]-interval[0]
            shift_len = (int)(random_shift_ratio*signal_len)
            new_start = max(0, interval[0]-np.random.randint(shift_len))
            new_end = min(len(signal), interval[1]+np.random.randint(shift_len))
            interval = [new_start, new_end]
        interval[1] = min(interval[1], interval[0]+origin_len)#cut too long
        out_signal = [signal_value for value_index, signal_value in \
            enumerate(signal[interval[0]:interval[1], 0]) if value_index%downsample_count==0]#need to choose the first channel
        return out_signal
            
    def _set_label(self, label, label_to_number_dict):#turn to number
        return label_to_number_dict[label]
    def _get_data(self, signal_dict, label_buffer, label_to_number_dict, origin_len, downsample_count):
        signal_list = []
        label_list = []
        for raw_label in label_buffer:
            label = raw_label[0]
            signal_name, signal_start, signal_end = raw_label[1]
            signal_list.append(self.set_signal(signal_dict[signal_name], [signal_start, signal_end], origin_len, downsample_count))
            label_list.append(self._set_label(label, label_to_number_dict))
        # max_len = max([len(signal) for signal in signal_list])
        max_len = (int)(np.ceil(origin_len/downsample_count))
        _inputs = np.zeros((len(signal_list),max_len))
        _masks = np.zeros((len(signal_list),max_len))
        for signal_index, signal in enumerate(signal_list):
            _inputs[signal_index][:len(signal)] = signal
            _masks[signal_index][:len(signal)] = np.ones_like(signal)
        _labels = np.stack(label_list, axis=0)
        return _inputs, _masks, _labels
    def show_data(self, test_label, label_list, origin_len, downsample_count, show_len=10, test_index=0):
        label_dict, signal_dict = self.data_read()
        label_to_number_dict = {label:label_index for label_index, label in enumerate(label_list)}
        index_buffer_dict = {label:[i for i in range(len(label_dict[label]))] for label in label_list}
        if self.shuffle:
            for index_key, index_list in index_buffer_dict.items():
                np.random.shuffle(index_list)
        test_label_buffer = [[test_label, label_dict[test_label][temp_index]] for temp_index \
            in index_buffer_dict[test_label][test_index:test_index+show_len]]
        test_inputs, test_masks, test_labels = self._get_data(signal_dict, \
            test_label_buffer, label_to_number_dict, origin_len, downsample_count)
        for test_index, test_input in enumerate(test_inputs):
            test_mask = test_masks[test_index]
            test_label = test_labels[test_index]
            plt.clf()
            print(test_label)
            index_list = [i for i in range(len(test_input))]
            pos_index_list = [index for index in index_list if test_mask[index]==1]
            pos_input = [test_input[pos_index] for pos_index in pos_index_list]
            plt.plot(pos_index_list, pos_input, color="blue")

            neg_index_list = [index for index in index_list if test_mask[index]==0]
            neg_input = [test_input[neg_index] for neg_index in neg_index_list]
            plt.plot(neg_index_list, neg_input, color="red")
            plt.show()
    def data_process(self, batch_size, label_list, origin_len, downsample_count, queue):
        label_dict, signal_dict = self.data_read()
        # print({label:len(label_dict[label]) for label in label_list})
        if self.uniform_class:
            start_index_dict = {label:0 for label in label_list}
            index_buffer_dict = {label:[i for i in range(len(label_dict[label]))] for label in label_list}
            if self.shuffle:
                for index_key, index_list in index_buffer_dict.items():
                    np.random.shuffle(index_list)
        else:
            start_index = 0
            label_set_list = []
            for label in label_list:
                label_set_list.extend([[label, label_set] for label_set in label_dict[label]])
            if self.shuffle:
                np.random.shuffle(label_set_list)
        class_size = batch_size//len(label_list)
        label_to_number_dict = {label:label_index for label_index, label in enumerate(label_list)}
        while 1:
            if queue.qsize() < 10:
                label_buffer = []
                if self.uniform_class:
                    for index_key, index_list in index_buffer_dict.items():
                        # index_key is label
                        start_index = start_index_dict[index_key]
                        temp_label_buffer = [[index_key,label_dict[index_key][temp_index]] for temp_index in index_list[start_index:start_index+class_size]]
                        label_buffer.extend(temp_label_buffer)
                else:
                    label_buffer = label_set_list[start_index:start_index+batch_size]
                queue.put(self._get_data(signal_dict, label_buffer, label_to_number_dict, origin_len, downsample_count))

                #adjust start index
                if self.uniform_class:
                    for index_key, start_index in start_index_dict.items():
                        start_index_dict[index_key] += class_size
                        if start_index_dict[index_key]+class_size > len(label_dict[index_key]):
                            start_index_dict[index_key] = 0
                            if self.shuffle:
                                np.random.shuffle(index_buffer_dict[index_key])
                else:
                    start_index += batch_size
                    if start_index+batch_size>len(label_set_list):
                        start_index = 0
                        if self.shuffle:
                            np.random.shuffle(label_set_list)
            else:
                time.sleep(0.1)

    def generator(self, batch_size, label_list, origin_len, downsample_count):
        '''
        can't use this func twice
        '''
        queue = Queue()
        now_process = Process(target=self.data_process, args=(batch_size, label_list, origin_len, downsample_count, queue), daemon=True)
        now_process.start()
        while True:
            if queue.empty() == False:
                yield queue.get()
    def one_class_picker(self, batch_size, test_label, label_list, origin_len, downsample_count):
        label_dict, signal_dict = self.data_read()
        start_index = 0
        label_set_list = [[test_label, label_set] for label_set in label_dict[test_label]]
        label_to_number_dict = {label:label_index for label_index, label in enumerate(label_list)}
        while 1:
            label_buffer = label_set_list[start_index:start_index+batch_size]
            yield self._get_data(signal_dict, label_buffer, label_to_number_dict, origin_len, downsample_count)
            start_index += batch_size
            if start_index+batch_size>len(label_set_list):
                break

if __name__ == "__main__":
    data_generator = SignalDataGenerator("val", shuffle=True)
    label_list = ["N","L","R","A","V"]
    origin_len = 420
    downsample_count = 6
    # data_generator.show_data("N", label_list, downsample_count)
    data_gen = data_generator.generator(100, label_list, origin_len, downsample_count)
    next(data_gen)