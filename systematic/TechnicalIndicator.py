from symtable import Class

import numpy as np
import pandas as pd
import os
from pathlib import Path

import matplotlib.pyplot as plt

class TechnicalIndicator:
    def __init__(self):
        pass

    def ADX(self, df, colName, highColName, lowColName, win):
        """
        Average Directional Index (ADX) is a technical analysis indicator used by some traders to determine the strength of a trend.
        The trend can be either up or down, and this is shown by two accompanying indicators, the Negative Directional Indicator (-DI) and the Positive Directional Indicator (+DI).
        """

        df['True Range'] = np.maximum(df[highColName] - df[lowColName],
                                      np.maximum(
                                          abs(df[highColName] - df[colName].shift()),
                                          abs(df[lowColName] - df[colName].shift())
                                        )
                                      )

        df['+DM'] = np.where((df[highColName] - df[highColName].shift()) > (df[lowColName].shift() - df[lowColName]),
                             np.maximum(df[highColName] - df[highColName].shift(), 0), 0)
        df['-DM'] = np.where((df[lowColName].shift() - df[lowColName]) > (df[highColName] - df[highColName].shift()),
                             np.maximum(df[lowColName].shift() - df[lowColName], 0), 0)

        df['+DI'] = 100 * (
                    df['+DM'].ewm(span=win, adjust=False).mean() / df['True Range'].ewm(span=win, adjust=False).mean())
        df['-DI'] = 100 * (
                    df['-DM'].ewm(span=win, adjust=False).mean() / df['True Range'].ewm(span=win, adjust=False).mean())

        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])

        indicatorName = f'{colName} {win}ADX'
        df[indicatorName] = df['DX'].ewm(span=win, adjust=False).mean()
        return indicatorName

    def RSI(self, df, colName, win, method='EMA'):
        """
        RelatiÎve Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
        RSI oscillates between zero and 100. Traditionally, and according to Wilder, RSI is considered overbought
        when above 70 and oversold when below 30.
        """

        def apply_func_decorator(func):
            prev_row = {}
            def wrapper(curr_row, **kwargs):
                val = func(curr_row, prev_row)
                prev_row.update(curr_row)
                return val
            return wrapper

        @apply_func_decorator
        def wilder_smoothing(curr_row, prev_row):
            if 'smoothed' not in prev_row:
                return curr_row['value']
            return (prev_row['smoothed'] * (win - 1) + curr_row['value']) / win

        diff = df[colName].diff()
        up = diff.apply(lambda x: np.nan if pd.isna(x) else max(0, x))
        dn = diff.apply(lambda x: np.nan if pd.isna(x) else -min(0, x))

        if method == 'EMA':
            # get rid of the first line of 0 or 100 RSI
            up = up.ewm(span=win, adjust=False).mean()
            up.loc[up.first_valid_index()] = np.nan
            dn = dn.ewm(span=win, adjust=False).mean()
            dn.loc[dn.first_valid_index()] = np.nan
        elif method == 'SMA':
            up = up.rolling(window=win).mean()
            dn = dn.rolling(window=win).mean()
        elif method == "Wilders":
            up = up.ewm(alpha=1 / win).mean()
            dn = dn.ewm(alpha=1 / win).mean()
        indicatorName = f'{colName} {win}RSI'
        df[indicatorName] = 100 - (100 / (1 + up / dn))
        return indicatorName
    
    def crossover(self, df, colName1, colName2):
        """
        Crossover is a trading signal that occurs when two moving averages cross each other.
        """
        indicatorName = f'{colName1}-{colName2} Crossover'
        df[f"{indicatorName} Prev"] = np.where(df[colName1].shift(1) < df[colName2].shift(1), 1, 0)
        df[indicatorName] = np.where(df[colName1] > df[colName2], 1, 0)
        df[indicatorName] *= df[f"{indicatorName} Prev"]
        return indicatorName

    def crossunder(self, df, colName1, colName2):
        """
        Crossunder is a trading signal that occurs when two moving averages cross each other.
        """
        indicatorName = f'{colName1}-{colName2} Crossunder'
        df[f"{indicatorName} Prev"] = np.where(df[colName1].shift(1) > df[colName2].shift(1), 1, 0)
        df[indicatorName] = np.where(df[colName1] < df[colName2], 1, 0)
        df[indicatorName] *= df[f"{indicatorName} Prev"]
        return indicatorName