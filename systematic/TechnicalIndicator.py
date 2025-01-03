from symtable import Class

import numpy as np
import pandas as pd
import os
from pathlib import Path
from utils import sign
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
        df.drop(columns=[f"{indicatorName} Prev"], inplace=True)
        return indicatorName

    def crossunder(self, df, colName1, colName2):
        """
        Crossunder is a trading signal that occurs when two moving averages cross each other.
        """
        indicatorName = f'{colName1}-{colName2} Crossunder'
        df[f"{indicatorName} Prev"] = np.where(df[colName1].shift(1) > df[colName2].shift(1), 1, 0)
        df[indicatorName] = np.where(df[colName1] < df[colName2], 1, 0)
        df[indicatorName] *= df[f"{indicatorName} Prev"]
        df.drop(columns=[f"{indicatorName} Prev"], inplace=True)
        return indicatorName

    def monthSeasonality(self, df, longMonList, shortMonList):
        """
        Month seasonality is a trading signal that occurs when the month is in the longMonList.
        """
        indicatorName = 'Month Seasonality'
        df.index = pd.to_datetime(df.index)
        df[indicatorName] = np.where(df.index.month.isin(longMonList), 1, np.where(df.index.month.isin(shortMonList), -1, 0))
        return indicatorName


    def sharpe(self, df, colName, win):
        indicatorName = f'{colName} {win}Sharpe'
        df[indicatorName] = df[colName].pct_change().rolling(window=win).mean() / df[colName].pct_change().rolling(window=win).std() * 252 ** 0.5
        return indicatorName

    def MA(self, df, colName, win, ema=False):
        indicatorName = f'{colName} {win}'
        if not ema:
            indicatorName += "SMA"
            df[indicatorName] = df[colName].rolling(window=win).mean()
        else:
            indicatorName += "EMA"
            df[indicatorName] = df[colName].ewm(span=win, adjust=False).mean()
        return indicatorName

    def MACD(self, df, colName, winFast, winSlow, win):
        indicatorName = f'{colName} MACD{winFast}-{winSlow}'
        df[f"{colName} MACD" ] = df[colName].ewm(span=winFast, adjust=False).mean() - df[colName].ewm(span=winSlow, adjust=False).mean()
        df[f"{colName} MACD-Signal" ] = df[f"{colName} MACD"].ewm(span=win, adjust=False).mean()
        co = self.crossover(df, f"{colName} MACD", f"{colName} MACD-Signal")
        cu = self.crossunder(df, f"{colName} MACD", f"{colName} MACD-Signal")

        df[f"{indicatorName}"] = df[co] - df[cu]
        # df["Flag Dummy"] = df[f"Flag"]
        # df.loc[df.index[0], f"Flag Dummy"] = np.nan
        # df[f"{indicatorName}"] = df["Flag Dummy"].replace(to_replace=0, method='ffill')
        return indicatorName

    def bollingerBand(self, df, colName, win, std):
        indicatorName = f'{colName} {win}BollingerBand'
        df[indicatorName] = df[colName].rolling(window=win).mean()
        df[f'{colName} {win}UpperBand'] = df[indicatorName] + std * df[colName].rolling(window=win).std()
        df[f'{colName} {win}LowerBand'] = df[indicatorName] - std * df[colName].rolling(window=win).std()
        return indicatorName

    def autoCorr(self, df, colName, win, lag):
        indicatorName = f'{colName} {win}Win {lag}AutoCorr'
        df[indicatorName] = df[colName].rolling(window=win).apply(lambda x: x.autocorr(lag))
        return indicatorName

    def syncSign(self, df, colName1, colName2, win):
        indicatorName = f'{colName1}-{colName2} {win}Sync'
        df[indicatorName] = np.where(df[colName1].pct_cahnge(win) * df[colName2].pct_change(win) > 0, 1, 0)
        return indicatorName

    def syncContinuousSign(self, df, colName1, colName2, win):
        indicatorName = f'{colName1}-{colName2} {win}SyncContinuous'
        df[indicatorName] = np.where(df[colName1].pct_cahnge() * df[colName2].pct_change() > 0, 1, 0)
        df[indicatorName] = df[indicatorName].rolling(window=win).sum().eq(win).astype(int)
        return indicatorName

    def leadSign(self, df, colName1, colName2, win):
        indicatorName = f'{colName1}-{colName2} {win}Lead'
        df[indicatorName] = np.where(df[colName1].pct_change(win) >= df[colName2].shift(win), 1, 2)
        return indicatorName