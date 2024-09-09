from symtable import Class

import numpy as np
import pandas as pd
import os
from pathlib import Path

import matplotlib.pyplot as plt
from pandas.errors import PerformanceWarning

from TechnicalIndicator import TechnicalIndicator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=PerformanceWarning)

class Signal:
    def __init__(self):
        pass

    def enterRSISignal(self, df, rsiCol, rsiInfList, rsiSupList):
        """
        RSI and ADX are two of the most powerful indicators. RSI is a momentum oscillator that measures the speed and change of price movements.
        ADX is a technical analysis indicator used by some traders to determine the strength of a trend.
        """
        stratNameList = []
        ti = TechnicalIndicator()
        for rsiInf in rsiInfList:
            for rsiSup in rsiSupList:
                stratName = f'enterRSI{rsiInf}-{rsiSup}'
                df[f'Inf{rsiInf}'] = rsiInf
                df[f'Sup{rsiSup}'] = rsiSup
                # Oversold
                rsi_inf_co = ti.crossover(df, rsiCol, f'Inf{rsiInf}')
                # Trend to sell
                rsi_inf_cu = ti.crossunder(df, rsiCol, f'Inf{rsiInf}')

                # Trend to buy
                rsi_sup_co = ti.crossover(df, rsiCol, f'Sup{rsiSup}')
                # Overbought
                rsi_sup_cu = ti.crossunder(df, rsiCol, f'Sup{rsiSup}')

                # Sell: From Sell Trend till Oversold
                df[f"{stratName} Sell"] = - df[rsi_inf_cu] + df[rsi_inf_co]
                mask = (df[f"{stratName} Sell"] == -1).cumsum() - (df[f"{stratName} Sell"] == 1).cumsum()
                df[f"{stratName} Sell"] = np.where((df[f"{stratName} Sell"] == 0) & (mask == 0), -1,
                                                  df[f"{stratName} Sell"])

                # Buy: From Buy Trend till Overbought
                df[f"{stratName} Buy"] = df[rsi_sup_co] - df[rsi_sup_cu]
                mask = (df[f"{stratName} Buy"] == 1).cumsum() - (df[f"{stratName} Buy"] == -1).cumsum()
                df[f"{stratName} Buy"] = np.where((df[f"{stratName} Buy"] == 0) & (mask == 0), 1, df[f"{stratName} Buy"])
                df[stratName] = df[f"{stratName} Sell"] + df[f"{stratName} Buy"]
                stratNameList.append(stratName)
        return stratNameList
    
    def exitRSISignal(self, df, rsiCol, rsiInfList, rsiSupList, buyStopList, sellStopList, momentum: bool = True):
        stratNameList = []
        ti = TechnicalIndicator()

        for rsiInf in rsiInfList:
            for rsiSup in rsiSupList:
                for buyStop in buyStopList:
                    for sellStop in sellStopList:
                        df["Buy Stop"] = buyStop
                        df["Sell Stop"] = sellStop
                        stratName = f'exitRSI Buy{rsiInf}-{buyStop} Sell{rsiSup}-{sellStop}'
                        df[f'Inf{rsiInf}'] = rsiInf
                        df[f'Sup{rsiSup}'] = rsiSup

                        # Oversold
                        rsi_inf_co = ti.crossover(df, rsiCol, f'Inf{rsiInf}')
                        # Stop Buy
                        rsi_stop_co = ti.crossover(df, rsiCol, 'Buy Stop')

                        # Overbought
                        rsi_sup_cu = ti.crossunder(df, rsiCol, f'Sup{rsiSup}')
                        # Stop Sell
                        rsi_stop_cu = ti.crossunder(df, rsiCol, 'Sell Stop')

                        # Sell: From Overbought till Stop Sell
                        df[f"{stratName} Sell"] = - df[rsi_sup_cu] + df[rsi_stop_cu]
                        df[f"{stratName} Sell"] = df[f"{stratName} Sell"].replace(to_replace=0, method='ffill')

                        # Buy: From Oversold till Stop Buy
                        df[f"{stratName} Buy"] = df[rsi_inf_co] - df[rsi_stop_co]
                        df[f"{stratName} Buy"] = df[f"{stratName} Buy"].replace(to_replace=0, method='ffill')
                        if momentum:
                            df[stratName] = - np.where(df[f"{stratName} Sell"] < 0, -1, np.where(df[f"{stratName} Buy"] > 0, 1, 0))
                        else:
                            df[stratName] = np.where(df[f"{stratName} Sell"] < 0, -1,
                                                       np.where(df[f"{stratName} Buy"] > 0, 1, 0))
                stratNameList.append(stratName)
        return stratNameList
    
    def enterRSIADXSignal(self, df, rsiCol, rsiInfList, rsiSupList, colName, colHigh, colLow, adxWinList, adxThreshList):
        """
        RSI and ADX are two of the most powerful indicators. RSI is a momentum oscillator that measures the speed and change of price movements.
        ADX is a technical analysis indicator used by some traders to determine the strength of a trend.
        """
        stratNameList = []
        ti = TechnicalIndicator()
        enterRSI = self.enterRSISignal(df, rsiCol, rsiInfList, rsiSupList)

        for r in enterRSI:
            for adxWin in adxWinList:
                for adxThresh in adxThreshList:
                    stratName = f'{r} ADX{adxWin}-{adxThresh}'
                    _adx = ti.ADX(df, colName, colHigh, colLow, adxWin)
                    df[stratName] = df[r] * np.where(df[_adx] > adxThresh, 1, 0)
                    stratNameList.append(stratName)
        return stratNameList


    def exitRSIADXSignal(self, df, rsiCol, rsiInfList, rsiSupList, colName, colHigh, colLow, adxWinList, adxThreshList):
        """
        RSI and ADX are two of the most powerful indicators. RSI is a momentum oscillator that measures the speed and change of price movements.
        ADX is a technical analysis indicator used by some traders to determine the strength of a trend.
        """
        stratNameList = []
        ti = TechnicalIndicator()
        enterRSI = self.exitRSISignal(df, rsiCol, rsiInfList, rsiSupList)

        for r in enterRSI:
            for adxWin in adxWinList:
                for adxThresh in adxThreshList:
                    stratName = f'{r} ADX{adxWin}-{adxThresh}'
                    _adx = ti.ADX(df, colName, colHigh, colLow, adxWin)
                    df[stratName] = df[r] * np.where(df[_adx] > adxThresh, 1, 0)
                    stratNameList.append(stratName)
        return stratNameList

    def fullRSISignal(self, df, rsiCol, enterInfList, enterSupList, exitInfList, exitSupList, momentum: bool = False):
        enterRSI = self.enterRSISignal(df, rsiCol, enterInfList, enterSupList)
        exitRSI = self.exitRSISignal(df, rsiCol, exitInfList, exitSupList)
        stratNameList = []
        for ent in enterRSI:
            for exi in exitRSI:
                stratName = f'{ent} {exi} [Mmt {momentum}]'
                if momentum:
                    df[stratName] = df[ent] - df[exi]
                else:
                    df[stratName] = df[ent] + df[exi]
                stratNameList.append(stratName)
        return stratNameList

if __name__ == "__main__":
    current_path = Path.cwd()
    root = current_path.parent
    df = pd.read_csv(os.path.join(root, 'data/raw/ndx-raw.csv'), index_col="Date" ,parse_dates=True)
    df.loc[: , ["NDX Index", "NDX Index High", "NDX Index Low"]] = df[[ "NDX Index", "NDX Index High", "NDX Index Low"]].shift(1)

    ti = TechnicalIndicator()
    ccy = 'KRW'
    _max = 0
    _maxName = ""
    for w in [7, 14, 21]:
        rsi = ti.RSI(df, f'NDX Index', w, method='EMA')
        # adx = ti.ADX(df, 'NDX Index', 'NDX Index High', 'NDX Index Low', 14)

        si = Signal()
        # stratNameList = si.exitRSISignal(df, rsi,range(5,20,2), range(70, 80, 2), [65, 70, 75], [35, 30, 25], momentum=True)
        # stratNameList = si.exitRSISignal(df, rsi,[10], [75], [75], [25])
        stratNameList = si.enterRSISignal(df, rsi,[10], [75])
        # stratNameList = si.exitRSIADXSignal(df, rsi, [42], [68], 'NDX Index',
        #                                  'NDX Index High', 'NDX Index Low', [14],
        #                                  [30])

        for stratName in stratNameList:
            df[f"{ccy} Curncy Ret"] = df[f"{ccy} Curncy"].pct_change()
            df[f"{stratName} Ret"] = df[stratName].shift(1) * df[f"{ccy} Curncy Ret"]
            df[f"{stratName} CumRet"] = df[f"{stratName} Ret"].cumsum()
            sharpe = df[f"{stratName} Ret"].mean() / df[f"{stratName} Ret"].std() * 252**0.5
            print(stratName, sharpe, df[f"{stratName} CumRet"].iloc[-1] * 1e4)
            if abs(sharpe) > _max:
                _max = abs(sharpe)
                _maxName = stratName + f" rsi{w}"
            df[f"{stratName} CumRet"].plot()
            plt.show()
    print(_maxName, _max)

