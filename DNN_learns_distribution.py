import tensorflow as tf
import numpy as np
from tensorflow import keras
import learning_utils as lu

lay = keras.layers

log_dir = "."  # directory to store everything
NOISE = 2.0  # large noise e.g. 20 // medium noise: 2 // very small noise 0.25

train_samples = 150000
test_samples = 10000


def epsilon(x):
    ''' Function that links Xmax to inputs.
    # Note: If you make the function way more complex you have to change improve the simple DNN and refine the training procedure in addition. '''
    return 0.9 * x**3 + 1.3 * x**2 - 2.45 * x + 0.34
    # return np.sin(x)


DEPENDENCY = np.random.randn(100)  # random dependency between Xmax and the respective observables

y_train = np.random.randn(train_samples)  # np.random.uniform(-3.5, 3.5, nsamples) #
y_train = lu.rand_gumbel(lgE=19, A=1, size=train_samples)
mu, std = y_train.mean(), y_train.std()
y_train = (y_train - 800) / 50  # rescaling

x_train = epsilon(DEPENDENCY + y_train[..., np.newaxis] + NOISE * np.random.randn(train_samples, 100))

y_test = np.random.randn(test_samples)  # y_test = np.random.uniform(-3.5, 3.5, nsamples)
y_test = lu.rand_gumbel(lgE=19, A=1, size=test_samples)
y_test = (y_test - 800) / 50  # rescaling

x_test = epsilon(DEPENDENCY + y_test[..., np.newaxis] + NOISE * np.random.randn(test_samples, 100))


def neural_network_model():
    ''' Use dopout if your model overfits'''
    input_ = lay.Input(shape=(100,))
    a = lay.Dense(128, activation="elu", kernel_initializer="he_normal")(input_)
    # a = lay.Dropout(0.3)(a)
    a = lay.Dense(128, activation="elu", kernel_initializer="he_normal")(a)
    # a = lay.Dropout(0.3)(a)
    a = lay.Dense(128, activation="elu", kernel_initializer="he_normal")(a)
    # a = lay.Dropout(0.3)(a)
    a = lay.Dense(128, activation="elu", kernel_initializer="he_normal")(a)
    # a = lay.Dropout(0.3)(a)
    output_ = lay.Dense(1, kernel_initializer="he_normal")(a)

    return keras.models.Model(input_, output_)


model = neural_network_model()

tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir + "/logs", update_freq="batch", write_graph=False)
file_writer_dep = tf.summary.create_file_writer(log_dir + "/logs" + "/dep")


def get_plotting_callback(model, x_test, y_test):
    def plot_dependency(epoch, logs):
        y_pr = model.predict(x_test).squeeze()
        lu.plot_performance(y_test, y_pr, "")
        image = lu.plot_to_image(fig)

        with file_writer_dep.as_default():
            tf.summary.image("Dependency Plot", image, step=epoch)

    return keras.callbacks.LambdaCallback(on_epoch_end=plot_dependency, on_train_begin=lambda x: plot_dependency(-1, x))


plotting_callback = get_plotting_callback(model, x_test, y_test)


model.compile(keras.optimizers.Adam(1E-3, decay=1E-4), loss="mean_squared_error")

y_pred = model.predict(x_test).squeeze()
reco = y_pred - y_test

fig, axes = lu.plot_performance(y_test, y_pred, name="before training NOISE: %.2f" % NOISE)
fig.savefig(log_dir + "/bias_test_before_training.pdf")

model.fit(x_train, y_train, batch_size=50, epochs=50, steps_per_epoch=1000, verbose=1, validation_split=0.2, callbacks=[tb, plotting_callback])

y_pred = model.predict(x_test).squeeze()
reco = y_pred - y_test

fig, axes = lu.plot_performance(y_test, y_pred, name="after training NOISE: %.2f" % NOISE)
fig.savefig(log_dir + "/bias_test.pdf", dpi=120)

y_pred_ = model.predict(x_train).squeeze()
fig, axes = lu.plot_performance(y_train, np.ones_like(y_pred_) * -10, name="after training NOISE: %.2f" % NOISE)
fig.savefig(log_dir + "/bias_train.pdf", dpi=120)

#####################################################
# OBSERVE THE TRAINING PROGRESS
# start TensorBoard using: tensorboard --samples_per_plugin images=1000 --logdir=.
# you can start TensorBoard usually via python /home/YOURUSERNAME/.local/lib/python3.6/site-packages/tensorboard/main.py'
# Open abritrary browser at http://127.0.1.1:6006/#images
