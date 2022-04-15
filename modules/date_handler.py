import pandas as pd


class DateHandler:

    def __init__(self):
        pass

    def date_to_day(self, df, col_name):
        df['day_of_week'] = df.apply(lambda row: pd.to_datetime(row[col_name]).day_name(), axis=1)
        df['n_day_of_week'] = df.apply(lambda row: pd.to_datetime(row[col_name]).day_of_week, axis=1)
        return df

    def date_to_month(self, df, col_name):
        df['month'] = df.apply(lambda row: pd.to_datetime(row[col_name]).month_name(), axis=1)
        return df

    def date_to_day_of_month(self, df, col_name):
        df['day_of_month'] = df.apply(lambda row: pd.to_datetime(row[col_name]).day, axis=1)
        return df

    def date_to_holiday(self, df):
        holidays = ['2018-01-01 00:00:00',
                    '2018-03-30 00:00:00',
                    '2018-04-02 00:00:00',
                    '2018-05-01 00:00:00',
                    '2018-05-21 00:00:00',
                    '2018-10-03 00:00:00',
                    '2018-10-31 00:00:00',
                    '2018-12-25 00:00:00',
                    '2018-12-26 00:00:00']
        df['holiday'] = df.apply(lambda row: 1 if str(row['time']) in holidays else 0, axis=1)
        # df['str_holiday'] = df.apply(lambda row: str(row['time']), axis=1)
        return df