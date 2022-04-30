from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import pandas as pd


class ItemModel:
    def __init__(self, inp_brand, inp_feature_1, inp_feature_2, inp_feature_3, inp_feature_4, inp_feature_5, out_item):
        self.__inp_brand = inp_brand
        self.__inp_feature_1 = inp_feature_1
        self.__inp_feature_2 = inp_feature_2
        self.__inp_feature_3 = inp_feature_3
        self.__inp_feature_4 = inp_feature_4
        self.__inp_feature_5 = inp_feature_5
        self.__out_item = out_item

        self.model = None

    def __define_input_variables(self):
        # an Input variable is made from every input array
        self.__input_brand = Input(shape=(self.__inp_brand.shape[1],), name='input_brand')
        self.__input_feature_1 = Input(shape=(self.__inp_feature_1.shape[1],), name='input_feat_1')
        self.__input_feature_2 = Input(shape=(self.__inp_feature_2.shape[1],), name='input_feat_2')
        self.__input_feature_3 = Input(shape=(self.__inp_feature_3.shape[1],), name='input_feat_3')
        self.__input_feature_4 = Input(shape=(self.__inp_feature_4.shape[1],), name='input_feat_4')
        self.__input_feature_5 = Input(shape=(self.__inp_feature_5.shape[1],), name='input_feat_5')

    def __define_output_variables(self):
        # all inputs were inserted into a dense layer with 5 units and 'relu' as activation function
        x1 = Dense(5, activation='relu')(self.__input_brand)
        x2 = Dense(5, activation='relu')(self.__input_feature_1)
        x3 = Dense(5, activation='relu')(self.__input_feature_2)
        x4 = Dense(5, activation='relu')(self.__input_feature_3)
        x5 = Dense(5, activation='relu')(self.__input_feature_4)
        x6 = Dense(5, activation='relu')(self.__input_feature_5)

        c = concatenate([x1, x2, x3, x4, x5, x6])  # all inputs are concatenated into one, mid-layer
        layer1 = Dense(64, activation='relu')(c)
        outputs = Dense(1, activation='sigmoid')(layer1)  # a single output is produced with value ranging between 0-1.
        return outputs

    def create_model(self):
        self.__define_input_variables()
        outputs = self.__define_output_variables()

        # create the model
        self.model = Model(
            inputs=[self.__input_brand,
                    self.__input_feature_1,
                    self.__input_feature_2,
                    self.__input_feature_3,
                    self.__input_feature_4,
                    self.__input_feature_5],
            outputs=outputs)

        return self.model

    def get_model_info(self):
        if self.model:
            self.model.summary()  # used to draw a summary(diagram) of the model
        else:
            print('Model is not defined!')

    def compiling(self):
        # while accuracy is used as a metrics here it will remain zero as this is no classification model
        optimizer = RMSprop(0.01)
        self.model.compile(loss='mean_squared_error',
                           optimizer=optimizer,      # 'adam'
                           metrics='acc'
                           )  # linear regression models are best gauged by their loss value

    def fit(self, inp_brand, inp_feature_1, inp_feature_2, inp_feature_3, inp_feature_4, inp_feature_5, out_item):
        # all the inputs were fed into the model and the training was completed
        history = self.model.fit(
            x=[inp_brand, inp_feature_1, inp_feature_2, inp_feature_3, inp_feature_4, inp_feature_5],
            y=out_item,
            batch_size=40,
            steps_per_epoch=200,
            epochs=1000,
            verbose=1,
            shuffle=False,
            validation_split=0.2)

    def format_input(self, inp_brand, inp_feature_1, inp_feature_2, inp_feature_3, inp_feature_4, inp_feature_5):
        inp_brand = np.array([inp_brand])
        inp_feature_1 = np.array([inp_feature_1])
        inp_feature_2 = np.array([inp_feature_2])
        inp_feature_3 = np.array([inp_feature_3])
        inp_feature_4 = np.array([inp_feature_4])
        inp_feature_5 = np.array([inp_feature_5])
        return inp_brand, inp_feature_1, inp_feature_2, inp_feature_3, inp_feature_4, inp_feature_5

    def format_output(self):
        out_item = np.array([self.__out_item])
        return out_item

    def testing(self, inp_brand, inp_feature_1, inp_feature_2, inp_feature_3, inp_feature_4, inp_feature_5, max_value):
        maxj = max_value
        inp_brand, inp_feature_1, inp_feature_2, inp_feature_3, inp_feature_4, inp_feature_5 = self.format_input(inp_brand, inp_feature_1, inp_feature_2, inp_feature_3, inp_feature_4, inp_feature_5)
        out_item = self.format_output()
        # predicting value for output
        result = self.model.predict([inp_brand,
                                     inp_feature_1,
                                     inp_feature_2,
                                     inp_feature_3,
                                     inp_feature_4,
                                     inp_feature_5])
        return result * maxj - 1
