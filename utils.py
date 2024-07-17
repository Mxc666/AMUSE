import numpy as np
import pandas as pd

def process_label(dt):

    len_column = len(dt.columns[:-1])


    label_series = dt.iloc[:, len_column]
    for i, per_ele in enumerate(list(set(label_series))):
        dt.loc[dt[len_column]==per_ele, len_column] = i
    dt = dt.rename(columns={len_column:'label'})
    dt = dt.reset_index(drop=True)

    return dt


def fill_data(data):
    
    X_data = data.iloc[:, :-1]
    X_data = X_data.replace('?', np.nan)
    
    num_df = X_data.select_dtypes(include='number')
    str_df = X_data.select_dtypes(include=object)    
    
    num_key = list(num_df.columns)  # 数值列名
    num_val = [num_df[num_name].mean() for num_name in num_key]
    num_dic = dict(map(lambda k,v:[k,v], num_key, num_val)) 
    
    str_key = list(str_df.columns)  # 字符列名
    str_val = [str_df[str_name].mode()[0] for str_name in str_key]
    str_dic = dict(map(lambda k,v:[k,v], str_key, str_val))
    
    fill_dic = {**num_dic, **str_dic} 
    
    
    data.fillna(value = fill_dic, inplace = True)
    return data


def str2int(data):
    
    str_df = data.select_dtypes(include=object)
    str_col = list(str_df.columns)
    for per_column in str_col:
        str_list = data[per_column].unique()
        int_list = [i for i in range(len(str_list))]
        str2int_dic = dict(map(lambda k,v:[k,v], str_list, int_list))
        data = data.replace({per_column: str2int_dic})
    
    return data


def standa_raw(Data):
    
    X_Data = Data.iloc[:, :-1]
    
    col_list = list(X_Data.columns)
    same_val = []
    for per_column in col_list:
        if len(X_Data[per_column].unique())==1:
            same_val.append(per_column)
    
    label_ = pd.DataFrame(Data.iloc[:, -1])
    X_Data = (X_Data-X_Data.min ())/ (X_Data.max ()-X_Data.min ()) 
    if same_val:
        for col in same_val:
            X_Data[col] = Data[col]
    return pd.concat([X_Data,label_],axis=1)


def output(Res_list, dataSize, name_):
    
    res_df = pd.DataFrame(Res_list)
    res_df.to_csv('Res/' + name_ + '_kFold_res.csv', index=False)
    
    mean_df = pd.DataFrame(res_df.mean())
    mean_df.rename(columns={0:name_}, inplace=True)  
    mean_df.loc['whole length'] = dataSize
    print("\n-------------- kinds of classifiers' mean val")
    print(mean_df)
    mean_df.T.to_csv('Res/' + name_ +'_average_kFold_res.csv')
    
     
def record_res(Metrics_dict, dataSize, ISMcc, name_):
    
    res = [] 
    for k_fold in list(Metrics_dict.keys()):
        
        curF_metric_dict = Metrics_dict[k_fold]  # NB_Acc, NB_MCC, SVM_Acc...
        res.append(curF_metric_dict)
    
    output(res, dataSize, name_)  # to csv
        

        
    
    
    