import pandas as pd
import os
from pathlib import Path

from DataMaster import DataMaster
from Metrics import Metrics
from Signal import Signal
from TechnicalIndicator import TechnicalIndicator
from Visualization import Visualization
from config import clientId, clientSecret

from datetime import date, datetime

from gs_quant.session import GsSession, Environment

import warnings
from pandas.errors import PerformanceWarning, SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=PerformanceWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def main():
    fromDate, toDate = date(2010, 1, 1), date.today()

    current_path = Path.cwd()
    root = current_path.parent
    if os.path.exists(os.path.join(root, 'data/raw/twd-points.csv')):
        df = pd.read_csv(os.path.join(root, 'data/raw/twd-points.csv'), parse_dates=['date'], index_col='date')
    else:
        GsSession.use(Environment.PROD, clientId, clientSecret, scopes=('read_product_data',))
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
        df["TWD1M Curncy"] = df["1M forwardPoint"] + df["spot"]

    df["TWD1M Curncy"] = df["1M forwardPoint"] + df["spot"]
    df.index.name = 'date'
    df.fillna(method='ffill', inplace=True)

    ti = TechnicalIndicator()
    si = Signal()
    me = Metrics()
    vi = Visualization()
    _max, _maxStrat, _maxMetrics, _metricsData = 0, '', [], dict()

    stratNameList = []
    # for win in [17]:
    #     for lag in range(2,30,3):
    ac1 = ti.autoCorr(df, 'TWD1M Curncy', 17, 2)
    ac2 = ti.autoCorr(df, 'TWD1M Curncy', 15, 9)
    buyStrat = si.TechTrendSignal(df, ac2, [0.9], [0.6], direction=1)[0]
    sellStrat = si.TechTrendSignal(df, ac1, [-0.7], [-0.7], direction=-1)[0]
    strat = f"{buyStrat} {sellStrat}"
    df[strat] = df[buyStrat] + df[sellStrat]
    stratNameList.append(strat)

    for strat in stratNameList:
        m = me.calcAllMetrics(df, "TWD1M Curncy", strat)
        _metricsData.update({strat: (m['Sharpe'], m['Hit'], m['Active'])})
        if m['Sharpe'] > _max:
            _max = m['Sharpe']
            _maxStrat = strat
            _maxMetrics = m

    print("##### Max Sharpe #####")
    print(_maxStrat, _max)
    print(sorted(_metricsData.items(), key=lambda x: x[1]))
    vi.plotStrat(df, _maxStrat, 'TWD1M Curncy', horizon=False, show=True, save=False)

    # current_path = Path.cwd()
    # root = current_path.parent
    # df = pd.read_csv(os.path.join(root, 'data/raw/ndx-raw.csv'), index_col="Date" ,parse_dates=True)
    # df.loc[: , ["NDX Index", "NDX Index High", "NDX Index Low"]] = df[[ "NDX Index", "NDX Index High", "NDX Index Low"]].shift(1)
    # df.index.name = 'date'
    #
    # ti = TechnicalIndicator()
    # me = Metrics()
    # si = Signal()
    # ccy = 'KRW'
    # _max = 0
    # _maxName = ""
    # for w in [7, 14, 21]:
    #     rsi = ti.RSI(df, f'NDX Index', w, method='EMA')
    #     # adx = ti.ADX(df, 'NDX Index', 'NDX Index High', 'NDX Index Low', 14)
    #
    #
    #     # stratNameList = si.exitRSISignal(df, rsi,range(5,20,2), range(70, 80, 2), [65, 70, 75], [35, 30, 25], momentum=True)
    #     # stratNameList = si.exitRSISignal(df, rsi,[10],  [75], [75], [25])
    #     # stratNameList = si.enterRSISignal(df, rsi,[10], [75])
    #     stratNameList = si.RSITrendSignal(df, rsi, range(5,95,10), range(5,95,10), -1)
    #
    #     for stratName in stratNameList:
    #         metrics = me.calcAllMetrics(df, 'KRW Curncy', stratName)
    #         print(metrics['Sharpe'])
    # # print(_maxName, _max)

if __name__ == '__main__':
    main()