


class Visualization:
    unit: str
    unitMultiplier: float
    diff: bool

    def __init__(self, bps: bool = True, diff: bool = False):
        self.unit = 'bps' if bps else 'pct'
        self.unitMultiplier = 1e4 if bps else 1e2
        self.diff = diff

    def plotCumRtn(self, df, signalColName):
        try:
            df[f"{signalColName} CumPnl({self.unit})"]
        except Exception as e:
            print(f'Please run metrics before plot the signal rtn')
            return None
        # Plot signal return
        ax = df[f"{signalColName} CumPnl({self.unit})"].plot()

        return ax

    def markSignal(self, df, instrumentColName):

    def save(self, path):
        # Save plot to path