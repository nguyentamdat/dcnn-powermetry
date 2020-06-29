import numpy as np
import tensorflow as tf
from functools import *
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt

data = np.load("data/batt_processed.npz", allow_pickle=True)
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

data_train = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(1000).batch(32)
data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class DCNN(Model):
    def __init__(self,):
        super(DCNN, self).__init__()
        self.conv1 = Conv2D(16, (1, 2), padding='same')
        self.conv2 = Conv2D(32, (3, 1), padding='same')
        self.conv3 = Conv2D(40, (3, 1), padding='same')
        self.conv4 = Conv2D(40, (3, 1), padding='same')
        self.conv5 = Conv2D(40, (3, 1), padding='same')
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
        self.fc1 = Dense(40)
        self.fc2 = Dense(40)
        self.fc3 = Dense(1)

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
        x = self.fc1(x)
        x = self.bn6(x, training=training)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn7(x, training=training)
        x = self.relu(x)
        return self.fc3(x)


model = DCNN()

loss_obj = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()
train_rmse = tf.keras.metrics.RootMeanSquaredError('train_rmse')
train_mae = tf.keras.metrics.MeanAbsoluteError('train_mae')

test_rmse = tf.keras.metrics.RootMeanSquaredError('test_rmse')
test_mae = tf.keras.metrics.MeanAbsoluteError('test_mae')


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


EPOCH = 50

# Keep results for plotting
train_rmse_results = []
train_mae_results = []

for epoch in range(EPOCH):
    train_rmse.reset_states()
    train_mae.reset_states()
    test_rmse.reset_states()
    test_mae.reset_states()

    for x, y in data_train:
        train_step(x, y)
    for t_x, t_y in data_test:
        test_step(t_x, t_y)
    train_rmse_results.append(train_rmse.result())
    train_mae_results.append(train_mae.result())
    template = 'Epoch {}, RMSE: {}, MAE: {}, Test RMSE: {}, Test MAE: {}'
    print(template.format(epoch+1, train_rmse.result(),
                          train_mae.result(), test_rmse.result(), test_mae.result()))

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("RMSE", fontsize=14)
axes[0].plot(train_rmse_results)

axes[1].set_ylabel("MAE", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_mae_results)
plt.show()
