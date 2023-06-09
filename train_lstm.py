from keras.layers.core import Dense
from keras.layers import Input, LSTM, Bidirectional
from keras.models import Model
from keras.optimizers import SGD, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import GRU
from keras import utils
import numpy as np
import time
import argparse
from keras.layers import InputSpec
from keras.layers import Layer
from matplotlib import pyplot as plt

class TemporalMaxPooling(Layer):
    """
    This pooling layer accepts the temporal sequence output by a recurrent layer
    and performs temporal pooling, looking at only the non-masked portion of the sequence.
    The pooling layer converts the entire variable-length hidden vector sequence
    into a single hidden vector.
    Modified from https://github.com/fchollet/keras/issues/2151 so code also
    works on tensorflow backend. Updated syntax to match Keras 2.0 spec.
    Args:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        3D tensor with shape: `(samples, steps, features)`.
        input shape: (nb_samples, nb_timesteps, nb_features)
        output shape: (nb_samples, nb_features)
    Examples:
        > x = Bidirectional(GRU(128, return_sequences=True))(x)
        > x = TemporalMaxPooling()(x)
    """

    def __init__(self, **kwargs):
        super(TemporalMaxPooling, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        if mask is None:
            mask = K.sum(K.ones_like(x), axis=-1)

        # if masked, set to large negative value so we ignore it
        # when taking max of the sequence
        # K.switch with tensorflow backend is less useful than Theano's
        mask = K.expand_dims(mask, axis=-1)
        mask = K.tile(mask, (1, 1, K.int_shape(x)[2]))
        masked_data = K.tf.where(
            K.equal(mask, K.zeros_like(mask)), K.ones_like(x) * -np.inf, x
        )  # if masked assume value is -inf
        return K.max(masked_data, axis=1)

    def compute_mask(self, input, mask):
        # do not pass the mask to the next layers
        return None

def lstm_model(train_data):
    main_input = Input(
        shape=(train_data.shape[1],
               train_data.shape[2]),
        name="main_input"
    )

    #headModel = LSTM(32)(main_input)

    headModel = Bidirectional(LSTM(256, return_sequences=True))(main_input)
    headModel = TemporalMaxPooling()(headModel)

    predictions = Dense(
        2,
        activation="sigmoid",
        kernel_initializer="he_uniform"
    )(headModel)
    model = Model(inputs=main_input, outputs=predictions)

    optimizer = Nadam(
        learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model

def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weights_save_name", type=str, default="lstm")
    args = parser.parse_args()

    # Training dataset loading
    train_data = np.load("lstm_40fpv_data.npy")
    train_label = np.load("lstm_40fpv_labels.npy")
    train_label = utils.to_categorical(train_label)
    print("Dataset Loaded...")

    # Train validation split
    trainX, valX, trainY, valY = train_test_split(
        train_data, train_label, shuffle=True, test_size=0.1
    )

    model = lstm_model(train_data)

    # Keras backend
    model_checkpoint = ModelCheckpoint(
        "trained_wts/" + args.weights_save_name + ".hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=0)

    print("Training is going to start in 3... 2... 1... ")

    # Model training
    H = model.fit(
        trainX,
        trainY,
        validation_data=(valX, valY),
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        callbacks=[model_checkpoint]#, stopping],
    )

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plots/training_plot.png")

    end = time.time()
    dur = end - start

    if dur < 60:
        print("Execution Time:", dur, "seconds")
    elif dur > 60 and dur < 3600:
        dur = dur / 60
        print("Execution Time:", dur, "minutes")
    else:
        dur = dur / (60 * 60)
        print("Execution Time:", dur, "hours")


if __name__ == '__main__':
    main()