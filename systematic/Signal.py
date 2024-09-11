from symtable import Class

import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import date, datetime

import matplotlib.pyplot as plt
from pandas.errors import PerformanceWarning

from TechnicalIndicator import TechnicalIndicator
from DataMaster import DataMaster
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

    def beforeMESignal(self, df, colName, BDList):
        stratNameList = []

        return stratNameList

    def afterMESignal(self, df, colName, BDList):
        stratNameList = []
        return  stratNameList

    def beforeIMMSignal(self, df, ccy, winList, momentum: bool = True):
        stratNameList = []

        fromDate = date(2010, 1, 1)
        toDate = date.today()
        dm = DataMaster()

        # Get IMM data
        _df = dm.getData(
            tickers=[f'USD{ccy}'],
            datasetName='FXFORWARDPOINTS_IMM',
            datasetArgs={
                'start': fromDate,
            },
            multiCol=True
        )

        fd = pd.pivot_table(_df, values='expirationDate', index='date', columns='expiration')
        fd.columns = [f"{col} fixingDate" for col in fd.columns]
        sd = pd.pivot_table(_df, values='settlementDate', index='date', columns='expiration')
        sd.columns = [f"{col} settlementDate" for col in sd.columns]
        pt = pd.pivot_table(_df, values='forwardPoint', index='date', columns='expiration')
        pt.columns = [f"{col} forwardPoint" for col in pt.columns]
        tmp = _df.loc[
            _df['expiration'] == 'IMM1', ['spot', 'expiration', 'forwardPoint', 'expirationDate', 'settlementDate',
                                          'spotSettlementDate']]
        imm = pd.concat([tmp, fd, sd, pt], axis=1).reset_index()
        imm['IMM1 fixingDate Shift'] = imm['IMM1 fixingDate'].shift(38)
        imm['distanceIMM1'] = [
            (r['IMM1 fixingDate Shift'] - r['date']).days if r['IMM1 fixingDate Shift'] >= r['date'] else -(
                        r['date'] - r['IMM1 fixingDate Shift']).days for i, r in imm.iterrows()]
        imm['distanceIMM1'] = [
            (r['IMM1 fixingDate'] - r['date']).days if pd.isna(r['distanceIMM1']) else r['distanceIMM1'] for i, r in
            imm.iterrows()]
        imm['IMM1-2'] = imm['IMM2 forwardPoint'] - imm['IMM1 forwardPoint']
        imm['IMM2-3'] = imm['IMM3 forwardPoint'] - imm['IMM2 forwardPoint']

        df = df.merge(imm, on='date', how='outer').fillna(method='ffill')
        for win in winList:
            stratName = f'{ccy} beforeIMM{win}'
            df[stratName] = np.where((df['distanceIMM1'] <= win) & (df['distanceIMM1'] > 0), 1, 0)
            df[stratName] *=  1 if momentum else -1
            stratNameList.append(stratName)

        return stratNameList

    def afterIMMSignal(self, df, ccy, winList, momentum: bool = True):
        stratNameList = []

        fromDate = date(2010, 1, 1)
        toDate = date.today()
        dm = DataMaster()

        # Get IMM data
        _df = dm.getData(
            tickers=[f'USD{ccy}'],
            datasetName='FXFORWARDPOINTS_IMM',
            datasetArgs={
                'start': fromDate,
            },
            multiCol=True
        )

        fd = pd.pivot_table(_df, values='expirationDate', index='date', columns='expiration')
        fd.columns = [f"{col} fixingDate" for col in fd.columns]
        sd = pd.pivot_table(_df, values='settlementDate', index='date', columns='expiration')
        sd.columns = [f"{col} settlementDate" for col in sd.columns]
        pt = pd.pivot_table(_df, values='forwardPoint', index='date', columns='expiration')
        pt.columns = [f"{col} forwardPoint" for col in pt.columns]
        tmp = _df.loc[
            _df['expiration'] == 'IMM1', ['spot', 'expiration', 'forwardPoint', 'expirationDate', 'settlementDate',
                                          'spotSettlementDate']]
        imm = pd.concat([tmp, fd, sd, pt], axis=1).reset_index()
        imm['IMM1 fixingDate Shift'] = imm['IMM1 fixingDate'].shift(38)
        imm['distanceIMM1'] = [
            (r['IMM1 fixingDate Shift'] - r['date']).days if r['IMM1 fixingDate Shift'] >= r['date'] else -(
                    r['date'] - r['IMM1 fixingDate Shift']).days for i, r in imm.iterrows()]
        imm['distanceIMM1'] = [
            (r['IMM1 fixingDate'] - r['date']).days if pd.isna(r['distanceIMM1']) else r['distanceIMM1'] for i, r in
            imm.iterrows()]
        imm['IMM1-2'] = imm['IMM2 forwardPoint'] - imm['IMM1 forwardPoint']
        imm['IMM2-3'] = imm['IMM3 forwardPoint'] - imm['IMM2 forwardPoint']

        df = df.merge(imm, on='date', how='outer').fillna(method='ffill')
        for win in winList:
            stratName = f'{ccy} afterIMM{win}'
            df[stratName] = np.where((abs(df['distanceIMM1']) <= win) & (df['distanceIMM1'] < 0), 1, 0)
            df[stratName] *= 1 if momentum else -1
            stratNameList.append(stratName)

        # df = df.merge(imm, on='date', how='outer')
        return stratNameList



if __name__ == "__main__":
    # current_path = Path.cwd()
    # root = current_path.parent
    # df = pd.read_csv(os.path.join(root, 'data/raw/ndx-raw.csv'), index_col="Date" ,parse_dates=True)
    # df.loc[: , ["NDX Index", "NDX Index High", "NDX Index Low"]] = df[[ "NDX Index", "NDX Index High", "NDX Index Low"]].shift(1)
    #
    # ti = TechnicalIndicator()
    # ccy = 'KRW'
    # _max = 0
    # _maxName = ""
    # for w in [7, 14, 21]:
    #     rsi = ti.RSI(df, f'NDX Index', w, method='EMA')
    #     # adx = ti.ADX(df, 'NDX Index', 'NDX Index High', 'NDX Index Low', 14)
    #
    #     si = Signal()
    #     # stratNameList = si.exitRSISignal(df, rsi,range(5,20,2), range(70, 80, 2), [65, 70, 75], [35, 30, 25], momentum=True)
    #     # stratNameList = si.exitRSISignal(df, rsi,[10], [75], [75], [25])
    #     stratNameList = si.enterRSISignal(df, rsi,[10], [75])
    #     # stratNameList = si.exitRSIADXSignal(df, rsi, [42], [68], 'NDX Index',
    #     #                                  'NDX Index High', 'NDX Index Low', [14],
    #     #                                  [30])
    #
    #     for stratName in stratNameList:
    #         df[f"{ccy} Curncy Ret"] = df[f"{ccy} Curncy"].pct_change()
    #         df[f"{stratName} Ret"] = df[stratName].shift(1) * df[f"{ccy} Curncy Ret"]
    #         df[f"{stratName} CumRet"] = df[f"{stratName} Ret"].cumsum()
    #         sharpe = df[f"{stratName} Ret"].mean() / df[f"{stratName} Ret"].std() * 252**0.5
    #         print(stratName, sharpe, df[f"{stratName} CumRet"].iloc[-1] * 1e4)
    #         if abs(sharpe) > _max:
    #             _max = abs(sharpe)
    #             _maxName = stratName + f" rsi{w}"
    #         df[f"{stratName} CumRet"].plot()
    #         plt.show()
    # print(_maxName, _max)

    current_path = Path.cwd()
    root = current_path.parent
    if os.path.exists(os.path.join(root, 'data/raw/twd-points.csv')):
        df = pd.read_csv(os.path.join(root, 'data/raw/twd-points.csv'))
    else:
        fromDate = date(2010, 1, 1)
        dm = DataMaster()
        ccy = 'TWD'

        _df1 = dm.getData(
            tickers=[
                f"FX Forward {ccy.upper()}/USD 1m"
            ],
            datasetName='FXFORWARDPOINTS_V2_PREMIUM',
            datasetArgs={
                'start': fromDate,
                'pricingLocation': 'HKG'
            },
            multiCol=True
        )
        _df3 = dm.getData(
            tickers=[
                f"FX Forward {ccy.upper()}/USD 3m"
            ],
            datasetName='FXFORWARDPOINTS_V2_PREMIUM',
            datasetArgs={
                'start': fromDate,
                'pricingLocation': 'HKG'
            },
            multiCol=True
        )

        _df1 = _df1.rename(columns={'fwdPoints': '1M forwardPoint'})
        _df3 = _df3.rename(columns={'fwdPoints': '3M forwardPoint'})
        df = pd.concat([_df1, _df3[['3M forwardPoint']]], axis=1)
        df[f"{ccy} 1x3 forwardPoint"] = df["3M forwardPoint"] - df["1M forwardPoint"]

        current_path = Path.cwd()
        root = current_path.parent
        df.to_csv(os.path.join(root, 'data/raw/twd-points.csv'))
    # %%
    si = Signal()
    beforeIMM = si.beforeIMMSignal(df, ccy='TWD', winList=[10], momentum=True)
    afterIMM = si.afterIMMSignal(df, ccy='TWD', winList=[10], momentum=False)

