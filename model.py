import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization, Dropout, Concatenate, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class Model:
    def _create_optimizer(self):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.init_lr,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        return Adam(learning_rate=lr_schedule, clipnorm=1.0)

    def compile_model(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def fit(self, X_time_series, X_selected_features, y_train_onehot, 
                    X_time_series_val, X_selected_features_val, y_val_onehot, 
                    y_train, batch_size=32, epochs=50):
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))

        history = self.model.fit(
            [X_time_series, X_selected_features], y_train_onehot,
            validation_data=([X_time_series_val, X_selected_features_val], y_val_onehot),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            class_weight=class_weight_dict
        )
        return history


class LSTMModel:
    def __init__(self, time_series_shape, n_selected_features, n_classes, init_lr=0.001):
        self.time_series_shape = time_series_shape
        self.n_selected_features = n_selected_features
        self.n_classes = n_classes
        self.init_lr = init_lr
        self.model = self._build_model()
        self.optimizer = self._create_optimizer()

    def _build_model(self):
        # 시계열 입력
        time_series_input = Input(shape=self.time_series_shape)
        lstm1 = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))(time_series_input)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(0.3)(lstm1)

        lstm2 = LSTM(64, kernel_regularizer=l2(0.01))(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(0.3)(lstm2)

        # 선택된 특성 입력
        selected_features_input = Input(shape=(self.n_selected_features,))
        dense1 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(selected_features_input)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)

        # 두 입력을 결합
        combined = Concatenate()([lstm2, dense1])
        combined = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.5)(combined)
        output = Dense(self.n_classes, activation='softmax')(combined)

        model = Model(inputs=[time_series_input, selected_features_input], outputs=output)
        return model

class CNNModel:
    def create_cnn_1d_model(time_series_shape, n_selected_features, n_classes):
        # 시계열 입력
        time_series_input = Input(shape=time_series_shape)
        conv1 = Conv1D(32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), padding='same')(time_series_input)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)

        conv2 = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)

        conv3 = Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = GlobalAveragePooling1D()(conv3)

        # 선택된 특성 입력
        selected_features_input = Input(shape=(n_selected_features,))
        dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(selected_features_input)
        dense1 = BatchNormalization()(dense1)

        # 두 입력을 결합
        combined = Concatenate()([conv3, dense1])

        dense2 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.5)(dense2)

        dense3 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dense2)
        dense3 = BatchNormalization()(dense3)
        dense3 = Dropout(0.5)(dense3)

        output = Dense(n_classes, activation='softmax')(dense3)

        model = Model(inputs=[time_series_input, selected_features_input], outputs=output)
        return model
    
class CNNLSTMModel:
    def create_simplified_model(time_series_shape, n_selected_features, n_classes):
        # 시계열 입력
        time_series_input = Input(shape=time_series_shape)
        conv1 = Conv1D(32, 3, activation='relu', kernel_regularizer=l2(0.01))(time_series_input)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(2)(conv1)

        lstm1 = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))(conv1)
        lstm1 = BatchNormalization()(lstm1)

        lstm2 = LSTM(64, kernel_regularizer=l2(0.01))(lstm1)
        lstm2 = BatchNormalization()(lstm2)

        # 선택된 특성 입력
        selected_features_input = Input(shape=(n_selected_features,))
        dense1 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(selected_features_input)
        dense1 = BatchNormalization()(dense1)

        # 두 입력을 결합
        combined = Concatenate()([lstm2, dense1])
        combined = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.5)(combined)
        output = Dense(n_classes, activation='softmax')(combined)

        model = Model(inputs=[time_series_input, selected_features_input], outputs=output)
        return model