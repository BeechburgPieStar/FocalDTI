import pandas as pd
import numpy as np
import os
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Conv2D, Flatten
from keras.layers.advanced_activations import PReLU, ReLU
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 3"

def perform_feature_scaling(x_train):
    """
    This method is used in order to perform feature scaling according to the 
    min-max scaler. The scaler can be replaced with another one, like the
    standard scaler 
    """
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train)

def perform_feature_selection(X_train, y_train, k_val):
    """ This method is used in order to perform a feature selection by selecting
    the best k_val features from X_train. It does so according to the chi2
    criterion. The method prints the chosen features and creates
    a new instance of X_train with only these features and returns it 
    """
    print("**********FEATURE SELECTION**********")
    # Create and fit selector
    selector = SelectKBest(chi2, k=k_val)
    selector.fit(X_train, y_train)
    #Get idxs of columns to keep
    idxs_selected = selector.get_support(indices=True)
    print(idxs_selected)
    X_new = SelectKBest(chi2, k=k_val).fit_transform(X_train, y_train)
    return X_new

def clear_missing_data(df):
    df_with_nan = df.replace("?", np.NaN)
    df_with_nan = df_with_nan.replace("unknown", np.NaN)
    return df_with_nan.dropna(0)

def is_dev(this_dev_name, dev_name):
    for i in range(10):
        if this_dev_name[i] == dev_name:
            return i

def get_is_dev_vec(this_dev_name, dev_names):
    return [is_dev(this_dev_name, dev_name) for dev_name in dev_names]

def catergorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p_t)^gamma)*log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    import keras.backend as K
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will no result in NaN
        # for o divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate weight that consists of modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss

    return focal_loss

y_col = 'device_category'
cols_to_drop = ['device_category']
use_cols = pd.read_csv(os.path.abspath('data/use_cols.csv'))

# Load Data
train = pd.read_csv(os.path.abspath('data/train.csv'), usecols=use_cols, low_memory=False)
validation = pd.read_csv(os.path.abspath('data/validation.csv'), usecols=use_cols, low_memory=False)
test = pd.read_csv(os.path.abspath('data/test.csv'), usecols=use_cols, low_memory=False)
devices = train[y_col].unique()

# Shuffle Data
train = shuffle(train)
validation = shuffle(validation)
test = shuffle(test)

# Remove missing data
train = clear_missing_data(train)
validation = clear_missing_data(validation)
test = clear_missing_data(test)

# Seperate data to features
x_train = train.drop(cols_to_drop, 1)
x_validation = validation.drop(cols_to_drop, 1)
x_test = test.drop(cols_to_drop, 1)

# Perform feature scaling for X
x_train = perform_feature_scaling(x_train)
x_validation = perform_feature_scaling(x_validation)
x_test = perform_feature_scaling(x_test)

y_train = pd.Series(get_is_dev_vec(devices, train[y_col]))
y_validation = pd.Series(get_is_dev_vec(devices, validation[y_col]))
y_test = pd.Series(get_is_dev_vec(devices, test[y_col]))

X_train = x_train[:, np.newaxis, :]
X_validation = x_validation[:, np.newaxis, :]
X_test = x_test[:, np.newaxis, :]

Y_train = keras.utils.to_categorical(y_train, num_classes=10)
Y_validation = keras.utils.to_categorical(y_validation, num_classes=10)
Y_test = keras.utils.to_categorical(y_test, num_classes=10)

in_shp = [1, 297]
dr = 0.5
model = Sequential()
model.add(Reshape((in_shp + [1]), input_shape=in_shp))
model.add(Conv2D(128, (1, 8), padding='valid', data_format="channels_last"))
model.add(ReLU())
model.add(Dropout(dr))
model.add(Conv2D(64, (1, 4), padding="valid", data_format="channels_last"))
model.add(ReLU())
model.add(Dropout(dr))
# model.add(Conv2D(32, (1, 4), padding="valid", data_format="channels_last"))
# model.add(ReLU())
# model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(dr))
model.add(Dense(128, activation='relu'))
model.add(Dropout(dr))
model.add(Dense(64, activation='relu'))
model.add(Dropout(dr))
model.add(Dense(10,  activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(),
#               metrics=['accuracy'])

model.compile(loss=[catergorical_focal_loss()],
              metrics=["accuracy"],
              optimizer=sgd)
model.summary()
model_path ="/data/hzm/IoT-device-type-identification-master/deep_learning/CNN/multiepoch"
filepath="model_{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(model_path, filepath), verbose=1, monitor='val_acc',
                             save_best_only=False)
tensorboard = TensorBoard("/data/hzm/IoT-device-type-identification-master/deep_learning/CNN/multiepoch/log", 0)
earlystopping = EarlyStopping(monitor="val_loss", patience=10, verbose=2, mode="auto")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto')

model.fit(X_train, Y_train, verbose=1, epochs=100, batch_size=128, validation_data=(X_validation, Y_validation),
          callbacks=[checkpoint, tensorboard, ])
model.save("/data/hzm/IoT-device-type-identification-master/deep_learning/CNN/multiepoch/cnn_100.hdf5")
score = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)
print('Total loss on Test Set:', score[0])
print('Accuracy of Testing Set:', score[1])