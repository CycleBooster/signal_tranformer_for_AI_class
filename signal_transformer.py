import tensorflow as tf
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Dropout, Dense

def get_angles(pos, depth, multiplier=1):
    pos = tf.cast(pos[:, tf.newaxis], float)
    depth = tf.cast(depth, float)
    depth_index_array = tf.range(depth)[tf.newaxis]
    angle_step = multiplier/(10000**(depth_index_array/(depth)))
    return pos * angle_step

def position_encoding(pos_shape, depth, x_base=0, y_base=0):#depth must be divisible by 2
    pos_map = tf.range(0, pos_shape, 1)
    pos_rads = get_angles(pos_map, depth//2)
    sines = tf.sin(pos_rads)
    cosines = tf.cos(pos_rads)

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = tf.cast(pos_encoding[tf.newaxis], tf.float32)#add batch and to float
    return pos_encoding

class SimplexAttentionLayer(Layer):
    def __init__(self, feature_depth, key_depth, head=8, regularizer=None):
        super().__init__()
        self.dst_layer_norm = LayerNormalization()
        self.src_layer_norm = LayerNormalization()
        self.src_to_dst_att = MultiHeadAttention(head, key_depth, value_dim=feature_depth, dropout=0.1, kernel_regularizer=regularizer)
    def call(self, dst_feature, src_feature, src_2_dst_mask=None, dst_pos_encoding=0, src_pos_encoding=0, training=False):
        dst_feature = self.dst_layer_norm(dst_feature)
        src_feature = self.src_layer_norm(src_feature)
        # print(dst_feature)
        # print(src_feature)
        # print()
        pos_dst_feature = dst_feature+dst_pos_encoding
        pos_src_feature = src_feature+src_pos_encoding
        #only q, k add position encoding
        out_feature, raw_att = self.src_to_dst_att(pos_dst_feature, src_feature, pos_src_feature, \
            attention_mask=src_2_dst_mask, return_attention_scores=True, training=training)#q, v, k, mask
            #mask_shape = (batch, q_len, k_len)
        # out_att = tf.reshape(raw_att, (tf.shape(raw_att)[0], -1, tf.shape(raw_att)[-1]))
        return out_feature, raw_att
class TransformerLayer(Layer):
    def __init__(self, feature_depth, key_depth, head=8, regularizer=None):
        super().__init__()
        self.att_layer = SimplexAttentionLayer(feature_depth, key_depth, head=head, regularizer=regularizer)

        self.dense_layer_norm = LayerNormalization()
        self.dense_temp_dense = Dense(feature_depth, "relu", kernel_regularizer=regularizer)
        self.dense_out_dense = Dense(feature_depth, kernel_regularizer=regularizer)
        self.dense_dropout = Dropout(0.1)
    def call(self, main_feature, key_feature, main_pos_encoding=0, key_pos_encoding=0, mask=None, training=False):
        temp_main_feature, temp_main_att = self.att_layer(main_feature, key_feature, \
            src_2_dst_mask=mask, dst_pos_encoding=main_pos_encoding, src_pos_encoding=key_pos_encoding, training=training)
        main_feature += temp_main_feature

        temp_main_feature = self.dense_layer_norm(main_feature)
        temp_main_feature = self.dense_temp_dense(temp_main_feature)
        temp_main_feature = self.dense_out_dense(temp_main_feature)
        temp_main_feature = self.dense_dropout(temp_main_feature, training=training)
        main_feature += temp_main_feature
        return main_feature


