import numpy as np
import pandas as pd
from datetime import date, datetime

from pandas.errors import PerformanceWarning, SettingWithCopyWarning

from TechnicalIndicator import TechnicalIndicator
from DataMaster import DataMaster
from utils import sign
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=PerformanceWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

class Signal:
    def __init__(self):
        pass

    def TechTrendSignal(self, df, techCol, techCOList, techCUList, direction=1):
        stratNameList = []
        ti = TechnicalIndicator()
        for co in techCOList:
            for cu in techCUList:
                stratName = f"{techCol}Trend{'Buy' if direction == 1 else 'Sell'} {techCol} CO{co} CU{cu}"
                df[f'{techCol}{co}'] = co
                df[f'{techCol}{cu}'] = cu
                rsi_co = ti.crossover(df, techCol, f'{techCol}{co}')
                rsi_cu = ti.crossunder(df, techCol, f'{techCol}{cu}')

                df[f"{stratName} Flag"] = - df[rsi_cu] + df[rsi_co]
                df[f"{stratName} Flag Dummy"] = df[f"{stratName} Flag"]
                df.loc[df.index[0], f"{stratName} Flag Dummy"] = np.nan
                df[f"{stratName} Flag Dummy"] = df[f"{stratName} Flag Dummy"].replace(to_replace=0, method='ffill')
                df[f"{stratName} Flag"] +=  df[f"{stratName} Flag Dummy"]

                if direction == -1:
                    df[f"{stratName}"] = (df[f"{stratName} Flag"] - 0.5) / 2
                    df[f"{stratName}"] = df[f"{stratName}"].apply(
                        lambda x: round(abs(x)) * sign(x) if not pd.isna(x) else np.nan
                    )
                    df[f"{stratName}"] = df[f"{stratName}"].replace(1, 0).replace(-2, -1)
                else:
                    df[f"{stratName}"] = (df[f"{stratName} Flag"] + 0.5) / 2
                    df[f"{stratName}"] = df[f"{stratName}"].apply(
                        lambda x: round(abs(x)) * sign(x) if not pd.isna(x) else np.nan
                    )
                    df[f"{stratName}"] = df[f"{stratName}"].replace(-1, 0).replace(2,1)
                stratNameList.append(stratName)
                df.drop(columns=[f"{techCol}{co}", f"{techCol}{cu}", f"{stratName} Flag", f"{stratName} Flag Dummy"], inplace=True)
        return stratNameList

    def enterRSISignal(self, df, rsiCol, rsiInfList, rsiSupList):
        """
        RSI and ADX are two of the most powerful indicators. RSI is a momentum oscillator that measures the speed and change of price movements.
        ADX is a technical analysis indicator used by some traders to determine the strength of a trend.
        """
        stratNameList = []
        ti = TechnicalIndicator()
        for rsiInf in rsiInfList:
            for rsiSup in rsiSupList:
                stratName = f'enterRSI {rsiCol} {rsiInf}-{rsiSup}'
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
                df[f"{stratName} SellFlag"] = - df[rsi_inf_cu] + df[rsi_inf_co]
                mask = (df[f"{stratName} SellFlag"] == -1).cumsum() - (df[f"{stratName} SellFlag"] == 1).cumsum()
                df[f"{stratName} Sell"] = np.where((df[f"{stratName} SellFlag"] == 0) & (mask == 0), -1,
                                                  df[f"{stratName} SellFlag"])
                df[f"{stratName} Sell"] = df[f"{stratName} Sell"].replace(1, 0)

                # Buy: From Buy Trend till Overbought
                df[f"{stratName} BuyFlag"] = df[rsi_sup_co] - df[rsi_sup_cu]
                mask = (df[f"{stratName} BuyFlag"] == 1).cumsum() - (df[f"{stratName} BuyFlag"] == -1).cumsum()
                df[f"{stratName} Buy"] = np.where((df[f"{stratName} BuyFlag"] == 0) & (mask > 0), 1, df[f"{stratName} BuyFlag"])
                df[f"{stratName} Buy"] = df[f"{stratName} Buy"].replace(-1, 0)
                df[stratName] = df[f"{stratName} Sell"] + df[f"{stratName} Buy"]
                df.drop(columns=[f'Inf{rsiInf}', f'Sup{rsiSup}', f"{stratName} SellFlag", f"{stratName} BuyFlag"], inplace=True)
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
                        stratName = f'exitRSI {rsiCol} Buy{rsiInf}-{buyStop} Sell{rsiSup}-{sellStop}'
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

    def DiveRSISignal(self, df, rsiCol):
        stratNameList = []
        stratName = f"DiveRSI {rsiCol}"
        df[f"{rsiCol} Diff"] = df[rsiCol].diff()
        df[stratName] = np.where(df[f"{rsiCol} Diff"] < 0, -1, np.where(df[f"{rsiCol} Diff"] > 0, 1, 0))
        df[stratName] = np.where(pd.isna(df[rsiCol]), np.nan, df[stratName])
        stratNameList.append(stratName)
        return stratNameList

    def DiveEnterRSISignal(self, df, rsiCol, rsiInfList, rsiSupList):
        stratNameList = self.enterRSISignal(df, rsiCol, rsiInfList, rsiSupList)
        diveRSIStrat = self.DiveRSISignal(df, rsiCol)[0]
        for stratName in stratNameList:
            df[stratName] = np.where(df[stratName] >0, df[stratName] * df[diveRSIStrat], np.where(df[stratName] < 0, -df[stratName] * df[diveRSIStrat], df[diveRSIStrat]))
        return stratNameList

    def RelativeEnterRSI(self, df, rsiCol, rsiInfList, rsiSupList, winList):
        # Calculate the rolling cumulative maximum
        win = 60


        stratNameList = []
        ti = TechnicalIndicator()
        for rsiInf in rsiInfList:
            for rsiSup in rsiSupList:
                stratName = f'enterRSI {rsiCol} {rsiInf}-{rsiSup}'
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
                df[f"{stratName} SellFlag"] = - df[rsi_inf_cu] + df[rsi_inf_co]
                mask = (df[f"{stratName} SellFlag"] == -1).cumsum() - (df[f"{stratName} SellFlag"] == 1).cumsum()
                df[f"{stratName} Sell"] = np.where((df[f"{stratName} SellFlag"] == 0) & (mask == 0), -1,
                                                   df[f"{stratName} SellFlag"])
                df[f"{stratName} Sell"] = df[f"{stratName} Sell"].replace(1, 0)
                # Relative low
                df[f'{rsiCol} {win}RollMim'] = df[rsiCol].rolling(window=win, min_periods=1).min()
                rsi_co = ti.crossunder(df, rsiCol, f'{rsiCol} {win}RollMin')

                # Buy: From Buy Trend till Overbought
                df[f"{stratName} BuyFlag"] = df[rsi_sup_co] - df[rsi_sup_cu]
                mask = (df[f"{stratName} BuyFlag"] == 1).cumsum() - (df[f"{stratName} BuyFlag"] == -1).cumsum()
                df[f"{stratName} Buy"] = np.where((df[f"{stratName} BuyFlag"] == 0) & (mask > 0), 1,
                                                  df[f"{stratName} BuyFlag"])
                df[f"{stratName} Buy"] = df[f"{stratName} Buy"].replace(-1, 0)
                # Relative max
                df[f'{rsiCol} {win}RollMax'] = df[rsiCol].rolling(window=win, min_periods=1).max()
                rsi_co = ti.crossover(df, rsiCol, f'{rsiCol} {win}RollMax')

                df[stratName] = df[f"{stratName} Sell"] + df[f"{stratName} Buy"]
                df.drop(columns=[f'Inf{rsiInf}', f'Sup{rsiSup}', f"{stratName} SellFlag", f"{stratName} BuyFlag"],
                        inplace=True)
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

    def AbsRtnRV(self, df, colName1, colName2, winList1, winList2, momentum=True):
        stratNameList = []
        for win1 in winList1:
            for win2 in winList2:
                stratName = f"Abs {colName1}{win1} {colName2}{win2} Mmt{momentum}"
                if momentum:
                    df[stratName] = np.where(df[colName1].pct_change(win1) > df[colName2].pct_change(win2), 1, -1)
                else:
                    df[stratName] = np.where(df[colName1].pct_change(win1) > df[colName2].pct_change(win2), -1, 1)
                df[stratName] = np.where((pd.isna(df[colName1].pct_change(win1))) | (pd.isna(df[colName2].pct_change(win2))), np.nan, df[stratName])
                stratNameList.append(stratName)
        return stratNameList

    def autoCorrTrend(self, df, colName, winList, lagList, trendCorrList, reverCorrList, momentum=True):
        stratNameList = []
        ti = TechnicalIndicator()
        df[f"{colName} Rtn"] = df[colName].pct_change()
        for win in winList:
            for lag in lagList:
                for tc in trendCorrList:
                    for rc in reverCorrList:
                        autoCorrColName = ti.autoCorr(df, f"{colName} Rtn", win, lag)
                        stratName = f'{colName} {win}Win AutoCorr{lag} TCorr{tc} RCorr{rc}'
                        df[f"{stratName} {win}win"] = df[colName].pct_change(win).apply(lambda x: sign(x))
                        if momentum:
                            _ = 1
                        else:
                            _ = -1
                        df[stratName] = np.where(df[autoCorrColName] > tc, _ * df[f"{stratName} {win}win"], np.where(df[autoCorrColName] < rc, -_ * df[f"{stratName} {win}win"], 0))
                        stratNameList.append(stratName)
        return stratNameList


    def autoCorrTrend2(self, df, colName, winList, lagList, rollWin, momentum=True):
        stratNameList = []
        ti = TechnicalIndicator()
        df[f"{colName} Rtn"] = df[colName].pct_change()
        for win in winList:
            for lag in lagList:
                autoCorrColName = ti.autoCorr(df, f"{colName} Rtn", win, lag)
                stratName = f'{colName} {win}Win AutoCorr{lag} TCorrRolling RCorrRolling'
                df[f"{stratName} {win}win"] = df[colName].pct_change(win).apply(lambda x: sign(x))
                if momentum:
                    _ = 1
                else:
                    _ = -1
                for rwin in rollWin:
                    df[f"{autoCorrColName} {rwin}min"] = -abs(df[f"{autoCorrColName}"].rolling(rwin).mean())
                    df[f"{autoCorrColName} {rwin}max"] = abs(df[f"{autoCorrColName}"].rolling(rwin).mean())
                    df[stratName] = np.where(df[autoCorrColName] > df[f"{autoCorrColName} {rwin}max"], _ * df[f"{stratName} {win}win"], np.where(df[autoCorrColName] < df[f"{autoCorrColName} {rwin}min"], -_ * df[f"{stratName} {win}win"], 0))
                    stratNameList.append(stratName)
        return stratNameList

    def pairLeadMASignal(self, df, colName1, colName2, win1List, win2List, rtnMA=False):
        stratNameList = []
        ti = TechnicalIndicator()
        for w1 in win1List:
            for w2 in win2List:
                stratName = f"{colName1} {w1}MA {colName2} {w2}MA"
                if rtnMA:
                    df[stratName] = np.where(df[colName1].pct_change().ewm(span=w1, adjust=False).mean() > df[colName2].pct_change().ewm(span=w2, adjust=False).mean(), 1, 0)
                else:
                    df[stratName] = np.where(df[colName1].pct_change(w1) > df[colName2].pct_change(w2), 1, 0)
                stratNameList.append(stratName)
        if f"{colName1}/{colName2}" not in df.columns:
            df[f"{colName1}/{colName2}"] = df[colName1] / df[colName2]
            df[f"{colName1}/{colName2}"] = df[f"{colName1}/{colName2}"].fillna(method='ffill')
        return stratNameList

    def MACDSignal(self, df, colName, winFastList, winSlowList, winSignalList):
        ti = TechnicalIndicator()
        stratNameList = []
        for wf in winFastList:
            for ws in winSlowList:
                for w in winSignalList:
                    stratName = f"{colName} MACD{wf}-{ws}-{w}"
                    macd = ti.MACD(df, colName, wf, ws, w)
                    # df[stratName] = df[macd].rolling(3).apply(lambda x: x.replace(to_replace=0, method='ffill')[-1])
                    df['Flag'] = df[macd]
                    df.loc[df.index[0], f"Flag"] = np.nan
                    df[stratName] = df["Flag"].replace(to_replace=0, method='ffill')
                    stratNameList.append(stratName)
        return stratNameList

if __name__ == "__main__":
    pass
