from dataclasses import dataclass
from datetime import datetime, date
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from fontTools.ttLib.tables.E_B_D_T_ import ebdt_bitmap_format_2


@dataclass
class Metrics:
    unit: str
    unitMultiplier: float
    diff: bool

    def __init__(self, bps: bool = True, diff: bool = False):
        self.unit = 'bps' if bps else 'pct'
        self.unitMultiplier = 1e4 if bps else 1e2
        self.diff = diff

    def calcDailyRtn(self, df, instrumentColName, signalColName):
        pnlColName = f"{signalColName} Pnl({self.unit})"
        if not self.diff:
            df[f"{instrumentColName} Rtn"] = df[instrumentColName].pct_change(1)
        else:
            df[f"{instrumentColName} Rtn"] = df[instrumentColName].diff(1)
        df[f"{instrumentColName} Rtn"] *= self.unitMultiplier
        df[pnlColName] = df[f"{instrumentColName} Rtn"] * df[signalColName].shift(1)
        return pnlColName


    def calcCumRtn(self, df, instrumentColName, signalColName):
        cumPnlColName = f"{signalColName} CumPnl({self.unit})"
        pnlColName = self.calcDailyRtn(df, instrumentColName, signalColName)
        df[cumPnlColName] = df[pnlColName].cumsum()
        return cumPnlColName


    def calcPeriodicPnl(self, df, instrumentColName, signalColName, period: str = 'YTD'):
        pnlColName = self.calcDailyRtn(df, instrumentColName, signalColName)
        periodicPnlColName = f"{signalColName} {period}Pnl({self.unit})"
        periodMap = {'YTD': 'A-DEC', 'MTD': 'M', 'QTD': 'Q', 'WTD': 'W'}
        df.reset_index(inplace=True)
        df[periodicPnlColName] = df[pnlColName].groupby([df.date.dt.to_period(periodMap[period])]).cumsum()
        df.set_index('date', inplace=True)
        return periodicPnlColName


    def calcHistYTDPnl(self, df, instrumentColName, signalColName):
        ytdPnlColName = self.calcPeriodicPnl(df, instrumentColName, signalColName, 'YTD')
        df.reset_index(inplace=True)
        histYTD = df[ytdPnlColName].groupby(df.date.dt.to_period('A-DEC')).last()
        df.set_index('date', inplace=True)
        histYTD.index = [i.year for i in histYTD.index]
        histYTD = histYTD.fillna(0)
        histYTD = ', '.join(":".join((str(k), f"{str(v)}{self.unit}")) for (k, v) in histYTD.to_dict().items())
        return histYTD

    def calcRecentMTDPnl(self, df, instrumentColName, signalColName):
        mtdPnlColName = self.calcPeriodicPnl(df, instrumentColName, signalColName, 'MTD')
        df.reset_index(inplace=True)
        recentMTD = df[mtdPnlColName].groupby(df.date.dt.to_period('M')).last()
        df.set_index('date', inplace=True)
        recentYear = recentMTD.index[-1].year
        recentMTD = recentMTD[recentMTD.index.year == recentYear]
        recentMTD.index = [i.strftime('%b') for i in recentMTD.index]
        recentMTD = recentMTD.fillna(0)
        recentMTD = ', '.join(":".join((str(k), f"{str(v)}{self.unit}")) for (k, v) in recentMTD.to_dict().items())
        return recentMTD

    def calcRecentQTDPnl(self, df, instrumentColName, signalColName):
        qtdPnlColName = self.calcPeriodicPnl(df, instrumentColName, signalColName, 'QTD')
        df.reset_index(inplace=True)
        recentQTD = df[qtdPnlColName].groupby(df.date.dt.to_period('Q')).last()
        recentDate = df.date.iloc[-1].date()
        df.set_index('date', inplace=True)
        yearStart = pd.Timestamp(datetime(recentDate.year, 1, 1))
        yearStart_str = f"{yearStart.year}Q{yearStart.quarter}"
        recentQTD = recentQTD[recentQTD.index >= yearStart_str]
        recentQTD = recentQTD.fillna(0)
        recentQTD = ', '.join(":".join((str(k), f"{str(v)}{self.unit}")) for (k, v) in recentQTD.to_dict().items())
        return recentQTD

    def calcRecentWTDPnl(self, df, instrumentColName, signalColName):
        wtdPnlColName = self.calcPeriodicPnl(df, instrumentColName, signalColName, 'WTD')
        df.reset_index(inplace=True)
        recentWTD = df[wtdPnlColName].groupby(df.date.dt.to_period('W')).last()
        recentDate = df.date.iloc[-1].date()
        df.set_index('date', inplace=True)
        weekStart = datetime.combine((recentDate - relativedelta(days=recentDate.weekday())), datetime.min.time())
        recentWTD.index = pd.to_datetime(recentWTD.index.astype(str).str.split('/').str[0])
        recentWTD = recentWTD[recentWTD.index >= weekStart]
        recentWTD = recentWTD.fillna(0)
        recentWTD = ', '.join(":".join((str(k), f"{str(v)}{self.unit}")) for (k, v) in recentWTD.to_dict().items())
        return recentWTD

    def calcDirectionHit(self, df, instrumentColName, signalColName, direction=1):
        hitNum = df.loc[(df[f"{instrumentColName} Rtn"] * df[signalColName].shift(1) > 0) & (df[signalColName].shift(1) == direction), signalColName].count()
        missNum = df.loc[(df[f"{instrumentColName} Rtn"] * df[signalColName].shift(1) < 0) & (df[signalColName].shift(1) == direction), signalColName].count()
        hit = hitNum / (hitNum + missNum)
        return hit

    def calcHit(self, df, instrumentColName, signalColName):
        hitNum = df.loc[(df[f"{instrumentColName} Rtn"] * df[signalColName].shift(1) > 0), signalColName].count()
        missNum = df.loc[(df[f"{instrumentColName} Rtn"] * df[signalColName].shift(1) < 0), signalColName].count()
        hit = hitNum / (hitNum + missNum)
        return hit

    def calcSharpe(self, df, instrumentColName, signalColName):
        pnlColName = self.calcDailyRtn(df, instrumentColName, signalColName)
        return df[pnlColName].mean() / df[pnlColName].std() * 252 ** 0.5 # std will skip NA by default

    def calcMDD(self, df, instrumentColName, signalColName):
        cumPnlColName = self.calcCumRtn(df, instrumentColName, signalColName)
        mddColName = f"{signalColName} MDD"
        df[f"{signalColName} cummax"] = df[cumPnlColName].cummax()
        df[mddColName] = (df[cumPnlColName] - df[f"{signalColName} cummax"]) / df[f"{signalColName} cummax"]
        mdd = df[mddColName].min()
        return mdd

    def calcMDDVol(self, df, instrumentColName, signalColName):
        pnlColName = self.calcDailyRtn(df, instrumentColName, signalColName)
        mdd = self.calcMDD(df, instrumentColName, signalColName)
        vol = df[pnlColName].std() * 252 **  0.5
        mddVol = mdd / vol
        return mddVol

    def calcActive(self, df, signalColName):
        total = df[signalColName].count()
        active = df[signalColName].apply(lambda x: 1 if not x==0 else 0).sum()
        return active / total

    def calcCorr(self, df, instrumentColName1, signalColName1, instrumentColName2, signalColName2):
        pnlColName1 = self.calcDailyRtn(df, instrumentColName1, signalColName1)
        pnlColName2 = self.calcDailyRtn(df, instrumentColName2, signalColName2)
        corr = df[pnlColName1].corr(df[pnlColName2])
        return corr

    def calcAllMetrics(self, df, instrumentColName1, signalColName1, instrumentColName2=None, signalColName2=None):
        metrics = {}
        metrics['Daily Rtn'] = df[self.calcDailyRtn(df, instrumentColName1, signalColName1)]
        metrics['Cum Rtn'] = df[self.calcCumRtn(df, instrumentColName1, signalColName1)]
        metrics['YTD Rtn'] = self.calcHistYTDPnl(df, instrumentColName1, signalColName1)
        metrics['WTD Rtn'] = self.calcRecentWTDPnl(df, instrumentColName1, signalColName1)
        metrics['MTD Rtn'] = self.calcRecentMTDPnl(df, instrumentColName1, signalColName1)
        metrics['QTD Rtn'] = self.calcRecentQTDPnl(df, instrumentColName1, signalColName1)
        metrics['Sharpe'] = self.calcSharpe(df, instrumentColName1, signalColName1)
        metrics['Hit'] = self.calcHit(df, instrumentColName1, signalColName1)
        metrics['Buy Hit'] = self.calcDirectionHit(df, instrumentColName1, signalColName1, direction=1)
        metrics['Sell Hit'] = self.calcDirectionHit(df, instrumentColName1, signalColName1, direction=-1)
        metrics['MDD'] = self.calcMDD(df, instrumentColName1, signalColName1)
        metrics['MDDVol'] = self.calcMDDVol(df, instrumentColName1, signalColName1)
        metrics['Active'] = self.calcActive(df, signalColName1)
        if instrumentColName2 is not None and signalColName2 is not None:
            metrics[f'Corr {signalColName1}-{signalColName2}'] = self.calcCorr(df, instrumentColName1, signalColName1, instrumentColName2, signalColName2)
        return metrics

if __name__ == '__main__':
    pass

