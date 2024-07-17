from os import path
import pandas as pd

def get_rawData(path, Isheader=None, IsID=True):
    raw_data = pd.read_csv(path, header=Isheader)
    if IsID:
        raw_data = raw_data.iloc[:, 1:]
        new_columnName = [i for i in range(len(list(raw_data.columns)))]
        raw_data.columns = new_columnName
    return raw_data