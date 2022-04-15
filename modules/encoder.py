from modules.date_handler import DateHandler as Date
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class Encoder:

    def __init__(self, code_type='one_hot'):
        self.__code_type = code_type
        self.__DateEncoder = None
        if self.__code_type == 'one_hot':
            self.__Encode = OneHotEncoder(handle_unknown='ignore')
        else:
            self.__Encode = None

        self.__Dates = Date()

    def date_encoder(self, df):
        """days of the month, months and day of the week will be available as one-hot-vector in a dataframe parameter

        :param df: containing time range
        :type df: dataframe
        :return: containing days of the month, months and day of the week
        :rtype: dataframe
        """
        time_start = df['date'][0]
        # set end time for encoder 2 months in the future for prognosis
        time_end = df['date'][df.shape[0] - 1] + pd.to_timedelta('2 M')
        date_range_df = pd.DataFrame({'time': pd.date_range(start=time_start, end=time_end, freq='D')})
        new_df = self.__Dates.date_to_day(date_range_df, col_name='date')
        new_df = self.__Dates.date_to_month(new_df, col_name='date')
        new_df = self.__Dates.date_to_day_of_month(new_df, col_name='date')
        # new_df = self.__Dates.date_to_holiday(new_df)
        encoder_df = new_df.join(self.one_hot_encode(new_df, col_name='day_of_week'))
        encoder_df = encoder_df.join(self.one_hot_encode(new_df, col_name='month'))
        encoder_df = encoder_df.join(self.one_hot_encode(new_df, col_name='day_of_month'))
        self.__DateEncoder = encoder_df
        return self.__DateEncoder

    def one_hot_encode(self, df, col_name):
        """encode column by using one hot encoding

        :param df: dataframe, should contain the column 'col_name'
        :type df: dataframe
        :param col_name: column name of the df that should be encoded
        :type col_name: str
        :return: old values from col_name column replaced with encoded column
        :rtype dataframe
        """
        # perform one-hot encoding on column
        encoder_df = pd.DataFrame(self.__Encode.fit_transform(df[[col_name]]).toarray())
        num_cols = encoder_df.shape[1]
        # convert columns to single vector, use previous col_name as suffix
        encoder_df['one_hot_' + col_name] = encoder_df.apply(lambda row: [row[i] for i in range(num_cols)], axis=1)
        # delete all columns except the column containing the one-hot-vectors
        encoder_df = encoder_df.drop([i_col for i_col in range(num_cols)], axis=1)
        return encoder_df
