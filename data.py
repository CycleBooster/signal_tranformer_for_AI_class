import json
import csv
import sys
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
class Progress():
    def __init__(self, data_len):
        self.data_len = data_len
        self.count = 0
        self.start_time = time.time()

    def update(self):
        self.count += 1
        print("\r%.2f%% " % (100*self.count/self.data_len), end=" ")
        print("%ds" % (time.time()-self.start_time), end=" ")
        print(str(self.count)+"/"+str(self.data_len), end="")
        if self.count == self.data_len:
            print()
def plt_cumsum(value_list, save_path, x_range=None, y_range=None, x_label=None, y_label=None, color="blue"):
    plt.clf()
    values, base = np.histogram(value_list, bins=40)
    cumsum_list = np.cumsum(values)
    if not isinstance(x_range, type(None)):
        plt.xlim(*x_range)
    if not isinstance(y_range, type(None)):
        plt.ylim(*y_range)
        # base = list(base)
        # base.insert(0, x_range[0]-(x_range[1]-x_range[0])*1e-5)#avoid no start point in cumsum
        # cumsum_list = list(cumsum_list)
        # cumsum_list.insert(0, y_range[0])#avoid no start point in cumsum
    if not isinstance(x_label, type(None)):
        plt.xlabel(x_label)
    if not isinstance(y_label, type(None)):
        plt.ylabel(y_label)
    plt.plot(base[:-1], cumsum_list, c=color)
    plt.savefig(save_path)
def show_image(img,name="test", wait=True):
    img = img.astype("uint8")
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey()
    else:
        cv2.waitKey(1)
def draw_label(image, point, label, label_color, font=cv2.FONT_ITALIC,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), label_color, cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
def show_result(imgs, preds, key_list, show=True, save=False, path="./save/", name="test_", wait=True):
    for img_index, img in enumerate(imgs):
        test_img = cv2.resize(img, (160,160))
        label = key_list[(int)(preds[img_index])]
        draw_label(test_img, (0, test_img.shape[1]-10), label, (0,0,0))
        if show:
            show_image(test_img, name=name, wait=wait)
        if save:
            if not os.path.isdir(path):
                os.makedirs(path)
            cv2.imwrite(path+name+str(img_index)+".jpg", test_img)
    return test_img
def show_attention(imgs, preds, atts, path="./save/"):
    #att.shape=(batch, head, class_num, H, W)
    if not os.path.isdir(path):
        os.makedirs(path)
    for img_index, img in enumerate(imgs):
        pred = preds[img_index]
        print(pred)
        att = atts[img_index]#(class_num, H, W)
        for att_index, raw_att in enumerate(att):
            raw_att = (raw_att/np.max(raw_att)*255).astype(np.uint8)
            cv2.imwrite(path+str(img_index)+"_att_"+str(att_index)+".jpg", cv2.resize(raw_att, (160,160)))

def load_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)
        except ValueError:
            print("error in ", data_path)
            sys.exit()
    return json_data
def json_write(json_data, json_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        try:
            f.write(json.dumps(json_data))
        except ValueError:
            print("error in ", json_path)
            sys.exit()
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
class MessageHandle():
    def __init__(self, data_path):
        self.data_path = data_path
        self.str_buffer = ""
    def add(self, message, interval=", "):
        self.str_buffer = self.str_buffer+message+interval
        print(message, end=interval)
    def refresh(self):
        self.str_buffer = ""
        print(end="\r")
    def save(self):
        self.str_buffer = self.str_buffer+"\n"
        with open(self.data_path, "a") as f:
            f.write(self.str_buffer)
        print()
        self.str_buffer = ""
class TxtWriter():
    def __init__(self, data_path):
        self.txt_file = open(data_path, "w")
    def __del__(self):
        self.txt_file.close()
    def write(self, data):#change line automatically
        self.txt_file.write(data)
class CsvReader():
    def __init__(self, data_path, delimiter=","):
        self.csvfile = open(data_path, newline='')
        self.reader = csv.reader(self.csvfile, delimiter=delimiter)
    def __del__(self):
        self.csvfile.close()
    def __iter__(self):
        return self
    def __next__(self):
        return next(self.reader)

class CsvWriter():
    def __init__(self, data_path, delimiter=","):
        self.csvfile = open(data_path, 'w', newline='')
        self.writer = csv.writer(self.csvfile, delimiter=delimiter)
    def __del__(self):
        self.csvfile.close()
    def write(self, raw_data):#write a line
        self.writer.writerow(raw_data)