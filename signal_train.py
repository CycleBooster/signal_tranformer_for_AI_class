import tensorflow as tf
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" #allow re-allocate memory in virtual memory. especially "triggered tf.function retracing"
import numpy as np
from signal_model import SignalTransformer
from signal_data_stream import SignalDataGenerator
from data import *
import time

class TrainHandle():
    def __init__(self, class_count, model_pretrain=None, input_size=81, model_name="Resnet50",\
        adjust_brightness=True, adjust_contrast=True, add_noise=True, adjust_size=True, random_flip=True,\
        min_crop_ratio=0.7):
        self.class_count = class_count
        # model
        print("set up model...")
        self.model = SignalTransformer(class_count, feature_depth=32, layer_count=2)

        # checkpoint
        self.model_path = "./model/"+model_name+"/"
        if not os.path.isdir(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))

        # loss and metric
        self.train_dict = self._get_compute_dict()
        self.train_dict["loss"]["loss"] = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_dict["loss reg"]["loss"] = tf.keras.metrics.Mean(name='train_loss')
        self.train_dict["acc"]["acc"] = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.val_dict = self._get_compute_dict()
        self.val_dict["loss"]["loss"] = tf.keras.losses.SparseCategoricalCrossentropy()
        self.val_dict["loss reg"]["loss"] = tf.keras.metrics.Mean(name='test_loss')
        self.val_dict["acc"]["acc"] = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        print("Setting completed.")
    def train(self, epochs, batch_size, train_data_handle, test_data_handle, \
        data_aug=True, start_epoch=0, lr=0.0001, save_dir='./train/new/'):
        # load weight if not from scratch
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self.model)
        ckpt_dir = save_dir+'ckpt/'
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        if start_epoch>0:
            print("load weight...")
            checkpoint.read(ckpt_dir+str(start_epoch-1))
        print("set data...")
        label_list = ["N","L","R","A","V"]
        origin_len = 420 #little larger than 360, the 1 sec.
        downsample_count = 6
        train_gen = train_data_handle.generator(batch_size, label_list, origin_len, downsample_count)
        train_steps = train_data_handle.get_data_len(label_list)//batch_size
        train_dict = dict(self.train_dict)
        # set optimizer
        train_dict["optimizer"] = optimizer

        print("train_steps:", train_steps)
        val_gen = test_data_handle.generator(batch_size, label_list, origin_len, downsample_count)
        val_steps = test_data_handle.get_data_len(label_list)//batch_size
        val_dict = dict(self.val_dict)
        # train
        print("start train")
        message_handle = MessageHandle(save_dir+"train.txt")
        for epoch in range(start_epoch, epochs):
            # Reset the metrics at the start of the next epoch
            for loss_key, train_loss in train_dict["loss reg"].items():
                train_loss.reset_states()
            for acc_key, train_acc in train_dict["acc"].items():
                train_acc.reset_states()
            for loss_key, test_loss in val_dict["loss reg"].items():
                test_loss.reset_states()
            for acc_key, test_acc in val_dict["acc"].items():
                test_acc.reset_states()

            start_time = time.time()
            # train
            for train_step in range(train_steps):
                train_signals, train_masks, train_labels =  next(train_gen)
                # add noise
                train_compute(train_labels, train_signals, self.model, train_dict, train_masks, training=True)

                message_handle.refresh()
                message_handle.add(
                    f'Epoch:{epoch} '
                    f'{(train_step+1)/train_steps:.1%} '
                    f'{(int)(time.time()-start_time)}s'
                    )
                for loss_key, train_loss in train_dict["loss reg"].items():
                    message_handle.add(f'{loss_key}: {train_loss.result():.4f}')
                for acc_key, train_acc in train_dict["acc"].items():
                    message_handle.add(f'{acc_key}: {train_acc.result():.1%}')
            message_handle.save()
            # validate
            for val_step in range(val_steps):
                val_signals, val_masks, val_labels = next(val_gen)
                test_compute(val_labels, val_signals, self.model, val_dict, val_masks)
                message_handle.add(f'{(val_step+1)/val_steps:.1%}')
                message_handle.refresh()
            for loss_key, test_loss in val_dict["loss reg"].items():
                message_handle.add(f'{loss_key}: {test_loss.result():.4f}')
            for acc_key, test_acc in val_dict["acc"].items():
                message_handle.add(f'{acc_key}: {test_acc.result():.1%}')
            message_handle.save()
            
            # save weight and test
            if epoch%5==4:
                # check point
                temp_save_dir = save_dir+str(epoch)+'/'
                if not os.path.isdir(temp_save_dir):
                    os.makedirs(temp_save_dir)
                checkpoint.write(ckpt_dir+str(epoch))
        # save model
        self.model.save(self.model_path)
    def test_data(self,  model_weight_path):
        # check point
        checkpoint = tf.train.Checkpoint(model=self.model)
        print("load weight...")
        checkpoint.read(model_weight_path).expect_partial()
        #set data
        val_dict = dict(self.val_dict)
        # label_list = ["N","L","R","A"]#,"V"]
        label_list = ["V"]
        origin_len = 420 #little larger than 360, the 1 sec.
        downsample_count = 6
        batch_size = 10
        test_data_handle = SignalDataGenerator("val", shuffle=True)
        for loss_key, test_loss in val_dict["loss reg"].items():
            test_loss.reset_states()
        for acc_key, test_acc in val_dict["acc"].items():
            test_acc.reset_states()
        val_gen = test_data_handle.generator(batch_size, label_list, origin_len, downsample_count)
        val_steps = test_data_handle.get_data_len(label_list)//batch_size
        print("val_steps=", val_steps)
        result_dict = {}
        for val_step in range(val_steps):
            val_signals, val_masks, val_labels = next(val_gen)
            #debug
            val_labels = [val_label+4 for val_label in val_labels]


            pred_list = test_compute(val_labels, val_signals, self.model, val_dict, val_masks)
            pred_list = pred_list[1].numpy()
            correct_list = [1 if val_label==pred_list[val_index] else 0 for val_index, val_label in enumerate(val_labels)]
            for val_index, val_label in enumerate(val_labels):
                correct_result = correct_list[val_index]
                if val_label not in result_dict:
                    result_dict[val_label] = [0,0]
                result_dict[val_label][correct_result] += 1
            print(
                f'{(val_step+1)/val_steps:.1%}', end="\r"
            )
        for loss_key, test_loss in val_dict["loss reg"].items():
            print(f'{loss_key}: {test_loss.result():.4f},', end=" ")
        for acc_key, test_acc in val_dict["acc"].items():
            print(f'{acc_key}: {test_acc.result():.1%},', end=" ")
        print()
        print(result_dict)
    def test_data_all_class(self,  model_weight_path):
        # check point
        checkpoint = tf.train.Checkpoint(model=self.model)
        print("load weight...")
        checkpoint.read(model_weight_path).expect_partial()
        #set data
        val_dict = dict(self.val_dict)
        label_list = ["N","L","R","A","V"]
        origin_len = 420 #little larger than 360, the 1 sec.
        downsample_count = 6
        batch_size = 10
        test_data_handle = SignalDataGenerator("val")
        result_dict = {}
        for test_label in label_list:
            for loss_key, test_loss in val_dict["loss reg"].items():
                test_loss.reset_states()
            for acc_key, test_acc in val_dict["acc"].items():
                test_acc.reset_states()
            print("test label is", test_label)
            val_gen = test_data_handle.one_class_picker(batch_size, test_label, label_list, origin_len, downsample_count)
            val_steps = test_data_handle.get_data_len([test_label])//batch_size
            print("val_steps=", val_steps)
            for val_step in range(val_steps):
                val_signals, val_masks, val_labels = next(val_gen)
                pred_list = test_compute(val_labels, val_signals, self.model, val_dict, val_masks)
                pred_list = pred_list[1].numpy()
                correct_list = [1 if val_label==pred_list[val_index] else 0 for val_index, val_label in enumerate(val_labels)]
                for val_index, val_label in enumerate(val_labels):
                    correct_result = correct_list[val_index]
                    if val_label not in result_dict:
                        result_dict[val_label] = [0,0]
                    result_dict[val_label][correct_result] += 1
                print(
                    f'{(val_step+1)/val_steps:.1%}', end="\r"
                )
            for loss_key, test_loss in val_dict["loss reg"].items():
                print(f'{loss_key}: {test_loss.result():.4f},', end=" ")
            for acc_key, test_acc in val_dict["acc"].items():
                print(f'{acc_key}: {test_acc.result():.1%},', end=" ")
            print()
        print(result_dict)
    def _get_compute_dict(self):
        #structure of compute_dict
        #dict in dict for "loss", "loss data weight", "loss reg", "acc", "acc data weight"
        #one callback in dict for "optimizer"
        #key = ["loss", "loss data weight", "optimizer", "loss reg", "acc", "acc data weight"]
        compute_dict = {
            "loss":{},
            "loss reg":{},
            "loss data weight":{},
            "loss sum weight":{},#default is 1
            "loss output index":{},#default is 0
            "acc":{},
            "acc data weight":{},
            "acc output index":{}#default is 0
        }
        return compute_dict

@tf.function
def train_compute(labels, inputs, model, compute_dict, mask, training=True):#, loss_object, optimizer, loss_metric, acc_metric):
    #structure of compute_dict
    #dict in dict for "loss", "loss data weight", "loss reg", "acc", "acc data weight"
    #one callback in dict for "optimizer"
    #key = ["loss", "loss data weight", "optimizer", "loss reg", "acc", "acc data weight"]
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        pred_list = model(inputs, mask=mask, training=training)
        loss_dict, total_loss = compute_loss(labels, pred_list, compute_dict)
    if "optimizer" in compute_dict:
        gradients = tape.gradient(total_loss, model.trainable_variables)
        compute_dict["optimizer"].apply_gradients(zip(gradients, model.trainable_variables))
    else:
        print("no optimizer in train compute.")
        sys.exit()
    # apply metric
    compute_loss_reg(loss_dict, compute_dict)
    compute_acc(labels, pred_list, compute_dict)

@tf.function
def test_compute(labels, inputs, model, compute_dict, mask, training=False):
    #structure of compute_dict
    #dict in dict for "loss", "loss data weight", "loss reg", "acc", "acc data weight"
    #key = ["loss", "loss data weight", "loss reg", "acc", "acc data weight"]

    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    pred_list = model(inputs, mask=mask, training=training)
    loss_dict, total_loss = compute_loss(labels, pred_list, compute_dict)
    # apply metric
    compute_loss_reg(loss_dict, compute_dict)
    compute_acc(labels, pred_list, compute_dict)
    return pred_list
@tf.function 
def tf_reg_compute(value_dict, reg_dict):
    for value_key, value in value_dict.items():
        reg_dict[value_key](value)

def compute_loss(labels, pred_list, compute_dict):
    loss_dict = {}
    for loss_key, loss_object in compute_dict["loss"].items():
        if loss_key in compute_dict["loss data weight"]:
            loss_data_weight = compute_dict["loss data weight"][loss_key]
        else:
            loss_data_weight = None
        if loss_key in compute_dict["loss output index"]:#choose output index, default 0
            loss_output_index = compute_dict["loss output index"][loss_key]
        else:
            loss_output_index = 0
        loss = loss_object(labels, pred_list[loss_output_index], sample_weight=loss_data_weight)
        loss_dict[loss_key] = loss
    #sum by "loss sum weight"
    out_loss = 0
    for loss_key, loss in loss_dict.items():
        if loss_key in compute_dict["loss sum weight"]:
            sum_weight = compute_dict["loss sum weight"][loss_key]
        else:
            sum_weight = 1
        out_loss += sum_weight*loss
    return loss_dict, out_loss
def compute_loss_reg(loss_dict, compute_dict):
    for loss_key, loss_reg in compute_dict["loss reg"].items():
        loss = loss_dict[loss_key]
        loss_reg(loss)#loss is scalar, no sample weight
def compute_acc(labels, pred_list, compute_dict):
    for acc_key, acc_metric in compute_dict["acc"].items():
        if acc_key in compute_dict["acc data weight"]:
            acc_data_weight = compute_dict["acc data weight"][acc_key]
        else:
            acc_data_weight = None
        if acc_key in compute_dict["acc output index"]:#choose output
            acc_output_index = compute_dict["acc output index"][acc_key]
        else:
            acc_output_index = 0
        acc_metric(labels, pred_list[acc_output_index], sample_weight=acc_data_weight)
if __name__ == "__main__":
    train_data_handle = SignalDataGenerator("train", random_shift=False, shuffle=True, uniform_class=False)
    test_data_handle = SignalDataGenerator("val")

    train_handle = TrainHandle(class_count=10, model_name="transformer")
    train_handle.train(epochs=30, batch_size=100, train_data_handle=train_data_handle, test_data_handle=test_data_handle, start_epoch=0)
    train_handle.test_data("./train/new/ckpt/29")
    train_handle.test_data_all_class("./train/new/ckpt/29")