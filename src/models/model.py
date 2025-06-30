from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import Model, Input, layers, backend as K
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Concatenate, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from config import HYPERPARAMS

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
                 lossfn='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(multi_label=False)]):
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
                lossfn='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(multi_label=False)]):
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
                lossfn='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(multi_label=False)]):
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
                lossfn='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(multi_label=False)]):
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

def binary_focal_loss(gamma=2., alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -alpha * K.pow(1. - pt, gamma) * K.log(pt)
        return K.mean(loss)
    return focal_loss

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads=4):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.query_dense = layers.Dense(d_model)
        self.key_dense = layers.Dense(d_model)
        self.value_dense = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        scaled_attention = tf.matmul(query, key, transpose_b=True)
        scaled_attention = scaled_attention / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(attention_weights, value)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        output = self.dense(output)
        return output

class BinaryCNNModel:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)
        
    def _build_model(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        
        # CNN layers
        x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Loss function 선택
        if HYPERPARAMS['focal_loss_gamma'] > 0:
            loss_fn = binary_focal_loss(
                gamma=HYPERPARAMS['focal_loss_gamma'],
                alpha=HYPERPARAMS['focal_loss_alpha']
            )
        else:
            loss_fn = 'binary_crossentropy'
            
        model.compile(
            optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return model
    
    def fit(self, x_train, y_train, x_val, y_val, class_weight=None):
        return self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=50,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
    
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    
    def predict(self, x):
        return (self.model.predict(x) > 0.5).astype(int)

class BinaryCNNAttentionModel:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)
        
    def _build_model(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        
        # First CNN block with residual connection
        x = layers.Conv1D(32, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(32, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # Skip connection
        x = layers.Add()([layers.Conv1D(32, 1, padding='same')(inputs), x])
        x = layers.MaxPooling1D(2)(x)
        
        # Second CNN block with residual connection
        skip = x
        x = layers.Conv1D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # Skip connection with 1x1 conv to match dimensions
        skip = layers.Conv1D(64, 1, padding='same')(skip)
        x = layers.Add()([skip, x])
        x = layers.MaxPooling1D(2)(x)
        
        # Third CNN block
        x = layers.Conv1D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Multi-head self attention with increased heads
        x = MultiHeadSelfAttention(d_model=128, num_heads=8)(x)  # Increased from 4 to 8 heads
        x = layers.BatchNormalization()(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers with reduced dropout
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)  # Reduced from 0.5
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)  # Reduced from 0.3
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Loss function 선택
        if HYPERPARAMS['focal_loss_gamma'] > 0:
            loss_fn = binary_focal_loss(
                gamma=HYPERPARAMS['focal_loss_gamma'],
                alpha=HYPERPARAMS['focal_loss_alpha']
            )
        else:
            loss_fn = 'binary_crossentropy'
            
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,  # Initial learning rate
            clipnorm=1.0  # Gradient clipping
        )
            
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return model
    
    def fit(self, x_train, y_train, x_val, y_val, class_weight=None):
        # Adjust class weights if provided
        if class_weight is not None and 1 in class_weight:
            class_weight[1] = min(class_weight[1], HYPERPARAMS['class_weight_max'])  # Use max value from config
            
        return self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=100,  # Increased epochs
            batch_size=16,  # Smaller batch size
            class_weight=class_weight,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,  # Increased from 10
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,  # Increased from 5
                    min_lr=1e-6,
                    verbose=1  # Learning rate 변경 시 출력
                )
            ]
        )
    
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    
    def predict(self, x):
        return (self.model.predict(x) > 0.5).astype(int)