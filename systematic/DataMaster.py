from dataclasses import dataclass
from gs_quant.session import GsSession, Environment
from gs_quant.data import Dataset
import config
from datetime import date, datetime

from config import clientId, clientSecret


@dataclass
class DataMaster:
    clientId: str
    clientSecret: str

    def __init__(self):
        self.clientId = clientId
        self.clientSecret = clientSecret

    def getAssetId(self, tickers: list, datasetName: str) -> dict:
        dataset = Dataset(datasetName)
        coverage = dataset.get_coverage(include_history=True)
        idMap = dict()
        inputName = "name"
        idName = "assetId"

        for t in tickers:
            if t in coverage[inputName].astype(str).values:
                idMap[t] = coverage[coverage[inputName].astype(str) == t][idName].values[0]
            else:
                print(f'missing the data from {datasetName} for {t}')
        if len(idMap) == 0:
            print(f'missing the data from {datasetName} for all tickers')
            return {}
        return (idMap, dataset)


    def getData(self, tickers, datasetName, datasetArgs, useAssetId: bool = False, multiCol: bool = False):
        try:
            GsSession.use(Environment.PROD, client_id=self.clientId, client_secret=self.clientSecret, scopes=('read_product_data',))
        except Exception as e:
            print(f'Error in authenticating with GS {e}')
            return None

        if not useAssetId:
            idMap, ds = self.getAssetId(tickers, datasetName)
            if len(idMap) == 0:
                return None
            idName = 'assetId'
            datasetArgs.update({idName: list(idMap.values())})
            df = ds.get_data(**datasetArgs).replace({idName: idMap})
        else:
            ds= Dataset(datasetName)
            df = ds.get_data(**datasetArgs)
        return df

if __name__ == '__main__':
    fromDate = date(2010,1,1)
    toDate = date.today()
    dm = DataMaster()
    df = dm.getData(
        tickers=['USDKRW'],
        datasetName='FXFORWARDPOINTS_IMM',
        datasetArgs={
            'start': fromDate,
        },
        multiCol=True
    )

    print(df)