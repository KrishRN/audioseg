# Script for training custom VAD model for the audioseg toolkit

import audioseg
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras import utils, models, layers
from tensorflow.keras.callbacks import ModelCheckpoint

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)
sess = tf.compat.v1.Session(config=session_conf)


# Define Model
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model=128, num_heads=1, causal=False, dropout=0.0):
        super().__init__()

        assert d_model % num_heads == 0
        depth = d_model // num_heads

        self.w_query = tf.keras.layers.Dense(d_model)
        self.split_reshape_query = tf.keras.layers.Reshape((-1, num_heads, depth))
        self.split_permute_query = tf.keras.layers.Permute((2, 1, 3))

        self.w_value = tf.keras.layers.Dense(d_model)
        self.split_reshape_value = tf.keras.layers.Reshape((-1, num_heads, depth))
        self.split_permute_value = tf.keras.layers.Permute((2, 1, 3))

        self.w_key = tf.keras.layers.Dense(d_model)
        self.split_reshape_key = tf.keras.layers.Reshape((-1, num_heads, depth))
        self.split_permute_key = tf.keras.layers.Permute((2, 1, 3))

        self.attention = tf.keras.layers.Attention(causal=causal, dropout=dropout)
        self.join_permute_attention = tf.keras.layers.Permute((2, 1, 3))
        self.join_reshape_attention = tf.keras.layers.Reshape((-1, d_model))

        self.dense = tf.keras.layers.Dense(d_model)

    def call(self, inputs, mask=None, training=None):
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v

        query = self.w_query(q)
        query = self.split_reshape_query(query)
        query = self.split_permute_query(query)

        value = self.w_value(v)
        value = self.split_reshape_value(value)
        value = self.split_permute_value(value)

        key = self.w_key(k)
        key = self.split_reshape_key(key)
        key = self.split_permute_key(key)

        if mask is not None:
            if mask[0] is not None:
                mask[0] = tf.keras.layers.Reshape((-1, 1))(mask[0])
                mask[0] = tf.keras.layers.Permute((2, 1))(mask[0])
            if mask[1] is not None:
                mask[1] = tf.keras.layers.Reshape((-1, 1))(mask[1])
                mask[1] = tf.keras.layers.Permute((2, 1))(mask[1])

        attention = self.attention([query, value, key], mask=mask)
        attention = self.join_permute_attention(attention)
        attention = self.join_reshape_attention(attention)

        x = self.dense(attention)

        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model=128, num_heads=1, dff=128, dropout=0.0):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout_attention = tf.keras.layers.Dropout(dropout)
        self.add_attention = tf.keras.layers.Add()
        self.layer_norm_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout_dense = tf.keras.layers.Dropout(dropout)
        self.add_dense = tf.keras.layers.Add()
        self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask=None, training=None):
        # print(mask)
        attention = self.multi_head_attention([inputs, inputs, inputs], mask=[mask, mask])
        attention = self.dropout_attention(attention, training=training)
        x = self.add_attention([inputs, attention])
        x = self.layer_norm_attention(x)
        # x = inputs

        ## Feed Forward
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense, training=training)
        x = self.add_dense([x, dense])
        x = self.layer_norm_dense(x)

        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers=1, d_model=128, num_heads=1, dff=128,
                 maximum_position_encoding=10000, dropout=0.0, **kwargs):
        super(Encoder, self).__init__()
        self.d_model = d_model
        super(Encoder, self).__init__(**kwargs)

        self.pos = positional_encoding(maximum_position_encoding, d_model)

        self.encoder_layers = [EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout=dropout) for _ in
                               range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(Encoder,self).get_config()
        config.update({"d_model": self.d_model})
        return config

    def call(self, inputs, mask=None, training=None):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(inputs)

        return x


def cnn_bilstm(output_layer_width):
    model = models.Sequential()
    model.add(layers.TimeDistributed(layers.Conv2D(64, (5, 5), activation='elu'), input_shape=(None, 32, 32, 1)))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='elu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='elu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.TimeDistributed(layers.Dense(128, activation='elu')))
    model.add(layers.Dropout(0.5))
    model.add(Encoder(num_layers=1, d_model=128, num_heads=1, dff=128, dropout=0.0))
    model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(output_layer_width, activation='softmax')))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Define training parameters
def train_model(model, x_train, y_train, validation_split, x_dev=None, y_dev=None, epochs=25, batch_size=64,
                callbacks=None):
    if validation_split:
        return model.fit(x_train[:, :, :, :, np.newaxis], y_train, validation_split=validation_split,
                         epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    elif x_dev is not None:
        return model.fit(x_train[:, :, :, :, np.newaxis], y_train,
                         validation_data=(x_dev[:, :, :, :, np.newaxis], y_dev),
                         epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    else:
        print('WARNING: no validation data, or validation split provided.')
        return model.fit(x_train[:, :, :, :, np.newaxis], y_train,
                         epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='CHPC_VAD_train.py',
                                     description='Train an instance of the audioseg VAD model.')

    parser.add_argument('-v', '--validation_dir', type=str,
                        help='a path to a Kaldi-style data directory containting \'wav.scp\', \'utt2spk\' and \'segments\'')

    parser.add_argument('-s', '--validation_split', type=float,
                        help='a percetage of the training data to be used as a validation set, if an explicit validation \
                              set is not defined using -v')

    parser.add_argument('train_dir', type=str,
                        help='a path to a Kaldi-style data directory containting \'wav.scp\', \'utt2spk\' and \'segments\'')

    parser.add_argument('model_name', type=str,
                        help='a filename for the model, the model will be saved as <model_name>.h5 in the output directory')

    parser.add_argument('out_dir', type=str,
                        help='a path to an output directory where the model will be saved as <model_name>.h5')

    args = parser.parse_args()

    # Fetch data
    data_train = audioseg.prep_labels.prep_data(args.train_dir)
    if args.validation_dir:
        data_dev = audioseg.prep_labels.prep_data(args.validation_dir)

    # Extract features
    feats_train = audioseg.extract_feats.extract(data_train)
    feats_train = audioseg.extract_feats.normalize(feats_train)
    if args.validation_dir:
        feats_dev = audioseg.extract_feats.extract(data_dev)
        feats_dev = audioseg.extract_feats.normalize(feats_dev)

    # Extract labels
    labels_train = audioseg.prep_labels.get_labels(data_train)
    labels_train['labels'] = audioseg.prep_labels.one_hot(labels_train['labels'])
    if args.validation_dir:
        labels_dev = audioseg.prep_labels.get_labels(data_dev)
        labels_dev['labels'] = audioseg.prep_labels.one_hot(labels_dev['labels'])

    # Train model
    X = audioseg.utils.time_distribute(np.vstack(feats_train['normalized-features']), 15)
    y = audioseg.utils.time_distribute(np.vstack(labels_train['labels']), 15)
    if args.validation_dir:
        X_dev = audioseg.utils.time_distribute(np.vstack(feats_dev['normalized-features']), 15)
        y_dev = audioseg.utils.time_distribute(np.vstack(labels_dev['labels']), 15)
    else:
        X_dev = None
        y_dev = None

    args.model_name
    checkpoint = ModelCheckpoint(filepath=f'{args.out_dir}/{args.model_name}.h5',
                                 save_weights_only=False, monitor='val_accuracy', mode='max', save_best_only=True)

    if y.shape[-1] == 2 or y.shape[-1] == 4:
        hist = train_model(cnn_bilstm(y.shape[-1]), X, y, args.validation_split, X_dev, y_dev, callbacks=[checkpoint])

        df = pd.DataFrame(hist.history)
        df.index.name = 'epoch'
        df.to_csv(f'{args.out_dir}/{args.model_name}_training_log.csv')
    else:
        print(
            f'ERROR: Number of classes {y.shape[-1]} is not equal to 2 or 4, see README for more info on using this training script.')
