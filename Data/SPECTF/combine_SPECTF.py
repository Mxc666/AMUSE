from numpy.lib import index_tricks
import pandas as pd

SPECTF_train = pd.read_csv('Data/SPECTF/SPECTF_train.csv', header=None)
SPECTF_test = pd.read_csv('Data/SPECTF/SPECTF_test.csv', header=None)

SPECTF_df = pd.concat([SPECTF_train, SPECTF_test], axis=0)
SPECTF_df = SPECTF_df.reset_index(drop=True)
len_ = len(list(SPECTF_df.columns))
SPECTF_df[len_] = SPECTF_df[0]
SPECTF_df = SPECTF_df.iloc[:, 1:]
new_columnName = [i for i in range(len(list(SPECTF_df.columns)))]
SPECTF_df.columns = new_columnName
print(SPECTF_df.head())
SPECTF_df.to_csv('Data/SPECTF/SPECTF.csv', index=False, header=False)