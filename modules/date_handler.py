import pandas as pd


class DateHandler:

    def __init__(self):
        pass

    def date_to_day(self, df, col_name):
        """add columns with day of the week information to input df
        0 monday, 1 tuesday, 2 wednesday, 3 thursday, 4 friday, 5 saturday, 6 sunday

        :param df: dataframe containing column containing timestamps
        :param col_name: name of the column containing timestamps
        :return: old df with new columns 'day_of_week' and 'n_day_of_week'
        """
        df['day_of_week'] = df.apply(lambda row: pd.to_datetime(row[col_name]).day_name(), axis=1)
        df['n_day_of_week'] = df.apply(lambda row: pd.to_datetime(row[col_name]).day_of_week, axis=1)
        return df

    def date_to_month(self, df, col_name):
        """add month for every timestamp in column to input df

        :param df: dataframe containing column containing timestamps
        :param col_name: name of the column containing timestamps
        :return: old df with new column 'month'
        """
        df['month'] = df.apply(lambda row: pd.to_datetime(row[col_name]).month_name(), axis=1)
        return df

    def date_to_day_of_month(self, df, col_name):
        """add day for every timestamp in column to input df

        :param df: dataframe containing column containing timestamps
        :param col_name: name of the column containing timestamps
        :return: old df with new column 'day_of_month'
        """
        df['day_of_month'] = df.apply(lambda row: pd.to_datetime(row[col_name]).day, axis=1)
        return df

    def date_to_holiday(self, df, col_name):
        """proof every timestamp of column whether it is a holiday (1) or not (0)

        :param df: dataframe containing column containing timestamps
        :param col_name: name of the column containing timestamps
        :return: old df with new column 'holiday'
        """
        # Todo: find holiday calender for germany 2020-2021
        holidays = []
        df['holiday'] = df.apply(lambda row: 1 if str(row[col_name]) in holidays else 0, axis=1)
        return df