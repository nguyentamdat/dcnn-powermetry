import numpy as np
import tensorflow as tf
from functools import *
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
import pandas as pd


def load_data():
    data = np.load("data/batt_processed_min.npz", allow_pickle=True)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    data_train = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(1000).batch(128)
    data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
    return data_train, data_test


data_train, data_test = load_data()


class DCNN(Model):
    def __init__(self, l2=tf.keras.regularizers.l2(0.0001)):
        super(DCNN, self).__init__()
        self.conv1 = Conv2D(16, (1, 2), padding='same', kernel_regularizer=l2)
        self.conv2 = Conv2D(32, (3, 1), padding='same', kernel_regularizer=l2)
        self.conv3 = Conv2D(40, (3, 1), padding='same', kernel_regularizer=l2)
        self.conv4 = Conv2D(40, (3, 1), padding='same', kernel_regularizer=l2)
        self.conv5 = Conv2D(40, (3, 1), padding='same', kernel_regularizer=l2)
        self.bn1 = BatchNormalization(1)
        self.bn2 = BatchNormalization(1)
        self.bn3 = BatchNormalization(1)
        self.bn4 = BatchNormalization(1)
        self.bn5 = BatchNormalization(1)
        self.bn6 = BatchNormalization(1)
        self.bn7 = BatchNormalization(1)
        self.relu = ReLU()
        self.maxpool = MaxPooling2D((3, 1), (2, 1))
        self.gavgpool = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.fc1 = Dense(40, kernel_regularizer=l2)
        self.fc2 = Dense(40, kernel_regularizer=l2)
        self.fc3 = Dense(1, kernel_regularizer=l2)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.relu(x)
        x = self.gavgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn6(x, training=training)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn7(x, training=training)
        x = self.relu(x)
        return self.fc3(x)


class MaxAbsError(tf.keras.metrics.Metric):
    def __init__(self, name="max_abs_error", **kwargs):
        super(MaxAbsError, self).__init__(name=name, **kwargs)
        self.mae = self.add_weight(name="mae", initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(tf.cast(y_true, tf.float32) -
                       tf.cast(y_pred, tf.float32))
        error = tf.reduce_max(error)
        self.mae.assign(error)

    def result(self):
        return self.mae

    def reset_states(self):
        self.mae.assign(0.0)


model = DCNN()

loss_obj = tf.keras.losses.MeanSquaredError()
lr = 0.01
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
train_rmse = tf.keras.metrics.RootMeanSquaredError('train_rmse')
train_mae = MaxAbsError('train_mae')

test_rmse = tf.keras.metrics.RootMeanSquaredError('test_rmse')
test_mae = MaxAbsError('test_mae')


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_obj(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_rmse(y, predictions)
    train_mae(y, predictions)


@tf.function
def test_step(x, y):
    predictions = model(x, training=False)
    t_loss = loss_obj(y, predictions)

    test_rmse(y, predictions)
    test_mae(y, predictions)


EPOCH = 35

# Keep results for plotting
train_rmse_results = []
train_mae_results = []
test_rmse_results = []
test_mae_results = []
patience = 7
for epoch in range(EPOCH):
    train_rmse.reset_states()
    train_mae.reset_states()
    test_rmse.reset_states()
    test_mae.reset_states()
    for x, y in data_train:
        train_step(x, y)
    for t_x, t_y in data_test:
        test_step(t_x, t_y)
    if len(test_rmse_results) > 0 and test_rmse.result() - test_rmse_results[-1] > 0.01:
        patience -= 1
        if patience <= 0:
            print("Early stop")
            break
    train_rmse_results.append(train_rmse.result().numpy())
    train_mae_results.append(train_mae.result().numpy())
    test_rmse_results.append(test_rmse.result().numpy())
    test_mae_results.append(test_mae.result().numpy())
    template = 'Epoch {}, RMSE: {}, MAE: {}, Test RMSE: {}, Test MAE: {}'
    print(template.format(epoch+1, train_rmse.result(),
                          train_mae.result(), test_rmse.result(), test_mae.result()))

print(model.summary())
history = pd.DataFrame({'Train_RMSE': train_rmse_results, 'Test_RMSE': test_rmse_results,
                        'Train_MAE': train_mae_results, 'Test_MAE': test_mae_results})
history.to_excel('DCNN_LOG.xlsx')

fig, axes = plt.subplots(4, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("RMSE", fontsize=14)
axes[0].plot(train_rmse_results)

axes[1].set_ylabel("MAE", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_mae_results)
axes[2].set_ylabel("Test RMSE", fontsize=14)
axes[2].plot(test_rmse_results)
axes[3].set_ylabel("Test MAE", fontsize=14)
axes[3].plot(test_mae_results)
plt.show()
