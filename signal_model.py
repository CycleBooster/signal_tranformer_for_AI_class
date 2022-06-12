import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from signal_transformer import *

class SignalTransformer(Model):
    def __init__(self, class_num, feature_depth=32, layer_count=4, weight_decay=1e-5):
        super().__init__()
        regularizer = tf.keras.regularizers.L2(weight_decay)
        self.feature_depth = feature_depth
        self.class_embedding = tf.Variable(tf.zeros((1, 1,feature_depth), dtype=tf.float32), trainable=True)#true or false for a class
        self.self_att_layer_list = [TransformerLayer(feature_depth, feature_depth, head=8, regularizer=regularizer) for i in range(layer_count)]
        self.class_att_layer_list = [TransformerLayer(feature_depth, feature_depth, head=8, regularizer=regularizer) for i in range(layer_count)]
        self.out_dense = Dense(class_num, kernel_regularizer=regularizer)
    def __input_encoding(self, input_signals, depth, multiplier=1):
        #input_signals shape = (batch, seq_len)
        #input_encoding shape = (batch, seq_len, depth)
        input_signals = tf.cast(input_signals[:, :, tf.newaxis], float)
        depth = tf.cast(depth//2, float)
        depth_index_array = tf.range(depth)[tf.newaxis, tf.newaxis]
        angle_step = multiplier/(10000**(depth_index_array/(depth)))
        pos_rads = input_signals * angle_step

        sines = tf.sin(pos_rads)
        cosines = tf.cos(pos_rads)
        input_encoding = tf.concat([sines, cosines], axis=-1)
        return input_encoding
    def call(self, inputs, mask, training=False):
        inputs = tf.cast(inputs, tf.float32)
        mask = mask[:, tf.newaxis]#origin mask_shape = (batch, seq_len)
        #input encoding
        input_encoding = self.__input_encoding(inputs, self.feature_depth)

        #position encoding
        pos_encoding = position_encoding(input_encoding.shape[1], self.feature_depth)
        #feature computing
        signal_feature = input_encoding
        class_feature = self.class_embedding*tf.ones_like(input_encoding[:, :1, :1])#broadcast batch
        for layer_index, self_att_layer in enumerate(self.self_att_layer_list):
            signal_feature = self_att_layer(signal_feature, signal_feature, \
                main_pos_encoding=pos_encoding, key_pos_encoding=pos_encoding, mask=mask, training=training)
            class_feature = self.class_att_layer_list[layer_index](class_feature, signal_feature, \
                key_pos_encoding=pos_encoding, mask=mask, training=training)
        #output
        class_feature = tf.squeeze(class_feature, axis=1)#to shape=(batch, feature_size)
        raw_y = self.out_dense(class_feature)
        out_y = tf.nn.softmax(raw_y)
        final_result = tf.argmax(out_y, axis=-1)
        return out_y, final_result, raw_y