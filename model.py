from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Concatenate, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class BaseModel(ABC):
    def __init__(self, optimizer, lossfn, metrics):
        self.optimizer = optimizer
        self.lossfn = lossfn
        self.metrics = metrics
        self.model = None 

    @abstractmethod
    def _build_model(self):
        pass

    def _compile_model(self):
        if self.model is None:
            raise ValueError("Model must be built before compiling.")

        if self.optimizer is None:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=1000,
                decay_rate=0.96,
                staircase=True
            )
            self.optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.lossfn,
            metrics=self.metrics
        )

    def fit(self, x, y_train_onehot, 
            x_val, y_val_onehot, 
            y_train, batch=32, epochs=50, es=None, cp=None, class_weight=None):
        '''
        x: input이 여러개이면 list형식으로 넣어주기.
        '''
        self._compile_model()

        if es is None:
            es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        if cp is None:
            cp = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')
        if class_weight is None:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight = dict(zip(np.unique(y_train), class_weights))

        history = self.model.fit(
            x=x, y=y_train_onehot,
            validation_data=(x_val, y_val_onehot),
            epochs=epochs,
            batch_size=batch,
            callbacks=[es, cp],
            class_weight=class_weight
        )
        return history
    
    def evaluate(self, x, y):
        '''
        x: input이 여러개이면 list형식으로 넣어주기.
        '''
        return self.model.evaluate(x, y)
    
    def predict(self, x):
        '''
        x: input이 여러개이면 list형식으로 넣어주기.
        
        '''
        y_pred = self.model.predict(x)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred




class LSTMModel(BaseModel):
    def __init__(self, x1_shape, x2_shape, n_classes, optimizer=None,
                 lossfn='categorical_crossentropy', metrics=['accuracy']):
        super().__init__(optimizer=optimizer, lossfn=lossfn, metrics=metrics)
        self.x1_shape = x1_shape
        self.x2_shape = x2_shape
        self.n_classes = n_classes
        self.model = self._build_model()


    def _build_model(self):
        # 시계열 입력
        x1 = Input(shape=self.x1_shape)
        lstm1 = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))(x1)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(0.3)(lstm1)

        lstm2 = LSTM(64, kernel_regularizer=l2(0.01))(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(0.3)(lstm2)

        # 선택된 특성 입력
        x2 = Input(shape=(self.x2_shape,))
        dense1 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x2)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)

        # 두 입력을 결합
        combined = Concatenate()([lstm2, dense1])
        combined = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.5)(combined)
        output = Dense(self.n_classes, activation='softmax')(combined)

        model = Model(inputs=[x1, x2], outputs=output)
        return model
    

class CNNModel(BaseModel):
    def __init__(self, x1_shape, x2_shape, n_classes, optimizer=None,
                lossfn='categorical_crossentropy', metrics=['accuracy']):
        super().__init__(optimizer=optimizer, lossfn=lossfn, metrics=metrics)
        self.x1_shape = x1_shape
        self.x2_shape = x2_shape
        self.n_classes = n_classes
        self.model = self._build_model()

    def _build_model(self):
        # 시계열 입력
        x1 = Input(shape=self.x1_shape)
        conv1 = Conv1D(32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), padding='same')(x1)
        conv1 = BatchNormalization()(conv1)
        # conv1 = MaxPooling1D(pool_size=2)(conv1)

        conv2 = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        # conv2 = MaxPooling1D(pool_size=2)(conv2)

        conv3 = Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = GlobalAveragePooling1D()(conv3)

        # 선택된 특성 입력
        x2 = Input(shape=self.x2_shape)
        dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x2)
        dense1 = BatchNormalization()(dense1)

        # 두 입력을 결합
        combined = Concatenate()([conv3, dense1])

        dense2 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.5)(dense2)

        dense3 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dense2)
        dense3 = BatchNormalization()(dense3)
        dense3 = Dropout(0.5)(dense3)

        output = Dense(self.n_classes, activation='softmax')(dense3)

        model = Model(inputs=[x1, x2], outputs=output)
        return model
    
class CNNx1OnlyModel(BaseModel):
    def __init__(self, x1_shape, n_classes, optimizer=None,
                lossfn='categorical_crossentropy', metrics=['accuracy']):
        super().__init__(optimizer=optimizer, lossfn=lossfn, metrics=metrics)
        self.x1_shape = x1_shape
        self.n_classes = n_classes
        self.model = self._build_model()

    def _build_model(self):
        # 시계열 입력
        x1 = Input(shape=self.x1_shape)
        conv1 = Conv1D(32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), padding='same')(x1)
        conv1 = BatchNormalization()(conv1)
        # conv1 = MaxPooling1D(pool_size=2)(conv1)

        conv2 = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        # conv2 = MaxPooling1D(pool_size=2)(conv2)

        conv3 = Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = GlobalAveragePooling1D()(conv3)

        dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(conv3)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.5)(dense1)

        dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.5)(dense2)

        output = Dense(self.n_classes, activation='softmax')(dense2)

        model = Model(inputs=x1, outputs=output)
        return model
    
class CNNLSTMModel(BaseModel):
    def __init__(self, x1_shape, x2_shape, n_classes, optimizer=None,
                lossfn='categorical_crossentropy', metrics=['accuracy']):
        super().__init__(optimizer=optimizer, lossfn=lossfn, metrics=metrics)
        self.x1_shape = x1_shape
        self.x2_shape = x2_shape
        self.n_classes = n_classes
        self.model = self._build_model()

    def _build_model(self):
        # 시계열 입력
        x1 = Input(shape=self.x1_shape)
        conv1 = Conv1D(32, 3, activation='relu', kernel_regularizer=l2(0.01))(x1)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(2)(conv1)

        lstm1 = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))(conv1)
        lstm1 = BatchNormalization()(lstm1)

        lstm2 = LSTM(64, kernel_regularizer=l2(0.01))(lstm1)
        lstm2 = BatchNormalization()(lstm2)

        # 선택된 특성 입력
        x2 = Input(shape=self.x2_shape)
        dense1 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x2)
        dense1 = BatchNormalization()(dense1)

        # 두 입력을 결합
        combined = Concatenate()([lstm2, dense1])
        combined = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.5)(combined)
        output = Dense(self.n_classes, activation='softmax')(combined)

        model = Model(inputs=[x1, x2], outputs=output)
        return model