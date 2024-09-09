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

    def __init__(self, bps: bool = True):
        self.unit = 'bps' if bps else 'pct'
        self.unitMultiplier = 1e4 if bps else 1e2

    def calcDailyRtn(self, df, instrumentColName, signalColName):
        """
        Calculate daily return for a given column
        """
        pnlColName = f"{signalColName} Pnl({self.unit})"
        df[f"{instrumentColName} Rtn"] = df[instrumentColName].pct_change(1) * self.unitMultiplier
        df[pnlColName] = df[f"{instrumentColName} Rtn"] * df[signalColName].shift(1)
        return pnlColName


    def calcCumRtn(self, df, instrumentColName, signalColName):
        """
        Calculate cumulative return for a given column
        """
        cumPnlColName = f"{signalColName} CumPnl({self.unit})"
        pnlColName = self.calcDailyRtn(df, instrumentColName, signalColName)
        df[cumPnlColName] = df[pnlColName].cumsum()
        return cumPnlColName


    def calcPeriodicPnl(self, df, instrumentColName, signalColName, period: str = 'YTD'):
        """
        Calculate periodic pnl for a given column
        """
        pnlColName = self.calcDailyRtn(df, instrumentColName, signalColName)
        periodicPnlColName = f"{signalColName} {period}Pnl({self.unit})"
        periodMap = {'YTD': 'A-DEC', 'M': 'MTD', 'Q': 'QTD', 'W': 'WTD'}
        df[periodicPnlColName] = df[pnlColName].groupby([df.date.dt.to_period(periodMap[period])]).cumsum()
        return periodicPnlColName


    def calcHistYTDPnl(self, df, instrumentColName, signalColName):
        ytdPnlColName = self.calcPeriodicPnl(df, instrumentColName, signalColName, 'YTD')
        histYTD = df[ytdPnlColName].groupby(df.date.dt.to_period('A-DEC')).last()
        histYTD.index = [i.year for i in histYTD.index]
        histYTD = histYTD.fillna(0)
        return histYTD

    def calcRecentMTDPnl(self, df, instrumentColName, signalColName):
        mtdPnlColName = self.calcPeriodicPnl(df, instrumentColName, signalColName, 'MTD')
        recentMTD = df[mtdPnlColName].groupby(df.date.dt.to_period('M')).last()
        recentYear = recentMTD.index[-1].year
        recentMTD = recentMTD[recentMTD.index.year == recentYear]
        recentMTD.index = [i.strftime('%b') for i in recentMTD.index]
        recentMTD = recentMTD.fillna(0)
        return recentMTD

    def calcRecentQTDPnl(self, df, instrumentColName, signalColName):
        qtdPnlColName = self.calcPeriodicPnl(df, instrumentColName, signalColName, 'QTD')
        recentQTD = df[qtdPnlColName].groupby(df.date.dt.to_period('Q')).last()
        recentDate = df.date.iloc[-1].date()
        yearStart = pd.Timestamp(datetime(recentDate.year, 1, 1))
        yearStart_str = f"{yearStart.year}Q{yearStart.quarter}"
        recentQTD = recentQTD[recentQTD.index >= yearStart_str]
        recentQTD = recentQTD.fillna(0)
        return recentQTD

    def calcRecentWTDPnl(self, df, instrumentColName, signalColName):
        wtdPnlColName = self.calcPeriodicPnl(df, instrumentColName, signalColName, 'WTD')
        recentWTD = df[wtdPnlColName].groupby(df.date.dt.to_period('W')).last()
        recentDate = df.date.iloc[-1].date()
        weekStart = datetime.combine((recentDate - relativedelta(days=recentDate.weekday())), datetime.min.time())
        recentWTD.index = pd.to_datetime(recentWTD.index.astype(str).str.split('/').str[0])
        recentWTD = recentWTD[recentWTD.index >= weekStart]
        recentWTD = recentWTD.fillna(0)
        return recentWTD

    def calcBuyHit(self, df, instrumentColName, signalColName):
        buyHit = np.sum((np.where(df[instrumentColName].shift(1) * df[signalColName].shift(1) == 1) & (df[signalColName] > 0), 1, 0))
        buyCount = np.sum(np.where(df[signalColName].shift(1) > 0, 1, 0))
        return buyHit / buyCount

    def calcSellHit(self, df, instrumentColName, signalColName):
        sellHit = np.sum((np.where(df[instrumentColName].shift(1) * df[signalColName].shift(1) == 1) & (df[signalColName] < 0), 1, 0))
        sellCount = np.sum(np.where(df[signalColName].shift(1) < 0, 1, 0))
        return sellHit / sellCount

    def calcSharpe(self, df, instrumentColName, signalColName):
        pnlColName = self.calcDailyRtn(df, instrumentColName, signalColName)
        return df[pnlColName].mean() / df[pnlColName].std() * 252 ** 0.5

    def calcCorr(self, df, instrumentColName1, signalColName1, instrumentColName2, signalColName2):
        pnlColName1 = self.calcDailyRtn(df, instrumentColName1, signalColName1)
        pnlColName2 = self.calcDailyRtn(df, instrumentColName2, signalColName2)
        corr = df[pnlColName1].corr(df[pnlColName2])
        return corr


if __name__ == '__main__':
    pass