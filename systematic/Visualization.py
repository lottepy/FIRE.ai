import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import date
import numpy as np
import pandas as pd
from dataclasses import dataclass

sys.path.insert(1, os.path.abspath(Path(os.getcwd()).parent))
sys.path.insert(1, os.path.abspath(Path(os.getcwd()).parent.parent))
sys.path.insert(1, os.path.join(os.path.abspath(Path(os.getcwd()).parent.parent), 'algo'))

@dataclass
class Visualization:
    unit: str
    unitMultiplier: float
    diff: bool
    def __init__(self, path: str = 'data/signal', bps: bool = True, diff: bool = False):
        self.unit = 'bps' if bps else 'pct'
        self.unitMultiplier = 1e4 if bps else 1e2
        self.diff = diff
        global picSavePath
        picSavePath = os.path.join(Path(__file__).parent.parent, path)
        if not os.path.exists(picSavePath):
            os.makedirs(picSavePath)

    def plotStrat(self, df, stratName, instrumentColName, horizon=True, show=True, save=False):
        fig, ax = plt.subplots(1,2,figsize=(20,5)) if horizon else plt.subplots(2,1,figsize=(10,10))
        fig.tight_layout(pad=5)
        ax1, ax3 = ax[0], ax[1]
        ax2, ax4 = ax1.twinx(), ax3.twinx()

        df['^'] = np.where(df[f"{stratName}"] > 0, df[instrumentColName], np.nan)
        df['v'] = np.where(df[f"{stratName}"] > 0, df[instrumentColName], np.nan)

        ytd = df.tail(int((date.today() - date(date.today().year, 1,1)).days))
        ytd[[instrumentColName]].plot(color='orange', label=instrumentColName, ax=ax1)
        ytd[[f"{stratName} YTDPnl({self.unit})"]].plot(ax=ax2, color='b', label=f'YTD Pnl({self.unit}')
        ytd[['^']].plot(marker='^', markersize=8, color='g', ax=ax1, linestyle='None')
        ytd[['v']].plot(marker='v', markersize=8, color='r', ax=ax1, linestyle='None')
        ax1.set_ylabel(instrumentColName)
        ax2.set_ylabel(f'YTD Pnl({self.unit})')
        ax1.grid(color='gray', linestyle='--', linewidth=0.5)
        ax2.grid(color='aqua', linestyle='--', linewidth=0.5)
        plt.xlabel(ytd.index.name)
        plt.title(stratName)
        ax1.legend([instrumentColName], loc='upper left')
        ax2.legend([f"YTD Pnl({self.unit})"], loc='lower left')

        df[[instrumentColName]].plot(color='orange', label=instrumentColName, ax=ax3)
        df[[f"{stratName} CumPnl({self.unit})"]].plot(ax=ax4, color='b', label=f'Cum Pnl({self.unit}')
        df[['^']].plot(marker='^', markersize=8, color='g', ax=ax3, linestyle='None')
        df[['v']].plot(marker='v', markersize=8, color='r', ax=ax3, linestyle='None')
        ax3.set_ylabel(instrumentColName)
        ax4.set_ylabel(f'Cum Pnl({self.unit})')
        ax3.grid(color='gray', linestyle='--', linewidth=0.5)
        ax4.grid(color='aqua', linestyle='--', linewidth=0.5)
        plt.xlabel(df.index.name)
        plt.title(stratName)
        ax3.legend([instrumentColName], loc='upper left')
        ax4.legend([f"Cum Pnl({self.unit})"], loc='lower left')

        if show:
            plt.show()
        picName = f"{stratName} {date.today().strftime('%Y%m%d')}.png".replace(" ", "").replace("|", "").replace("%", "")
        if save:
            saveDir = os.path.join(picSavePath, picName)
            fig.savefig(saveDir, bbox_inches='tight')
            return saveDir, picName

if __name__ == '__main__':
    vi = Visualization()