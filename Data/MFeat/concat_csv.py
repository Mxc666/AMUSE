import pandas as pd
import numpy as np

def add_label(df):

    label_list = []
    cur_i = 0
    for i in range(len(df)):
        if i%200 == 0:
            cur_i += 1
        label_list.append(cur_i)

    df['label'] = label_list
    return df



path_1 = 'Data/MFeat/mfeat-fac'
path_2 = 'Data/MFeat/mfeat-fou'
path_3 = 'Data/MFeat/mfeat-kar'
path_4 = 'Data/MFeat/mfeat-mor'
path_5 = 'Data/MFeat/mfeat-pix'
path_6 = 'Data/MFeat/mfeat-zer'

df_1 = pd.read_csv(path_1, header=None, delim_whitespace=True)
df_2 = pd.read_csv(path_2, header=None, delim_whitespace=True)
df_3 = pd.read_csv(path_3, header=None, delim_whitespace=True)
df_4 = pd.read_csv(path_4, header=None, delim_whitespace=True)
df_5 = pd.read_csv(path_5, header=None, delim_whitespace=True)
df_6 = pd.read_csv(path_6, header=None, delim_whitespace=True)
df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6], axis=1)

new_columnName = [i for i in range(len(list(df.columns)))]
df.columns = new_columnName
df = add_label(df)


df.to_csv('Data/MFeat/MFeat.csv', index=False)