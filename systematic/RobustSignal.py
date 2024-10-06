from attr import dataclass

from Signal import Signal
import numpy as np
import pandas as pd

@dataclass
class RobustSignal(Signal):


    def AbsRV(self, m1, m2, colName1, colName2, momentum=True):
        stratName = f"AbsRV {colName1} {colName2} Mmt{momentum}"
        if momentum:
            signal = np.where((pd.isna(m1)) | (pd.isna(m2)), np.nan, np.where(m1 > m2, 1, -1))
        else:
            signal = np.where((pd.isna(m1)) | (pd.isna(m2)), np.nan, np.where(m1 > m2, -1, 1))
        return signal, stratName