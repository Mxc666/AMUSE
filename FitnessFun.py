import numpy as np
from numpy.core.defchararray import join
from numpy.core.fromnumeric import _squeeze_dispatcher
import math
import pandas as pd
from scipy.spatial import distance_matrix

def updateSet(XcrossV):

    SF = []
    NSF = []
        
    mean_val = np.mean(XcrossV)
    n = len(XcrossV)
    for j in range(n):
        if XcrossV[j] >= mean_val:
            SF.append(j)
        else:
            NSF.append(j)
        
    return SF, NSF


def normalization_arr(res):
    max_res = np.max(res)
    min_res = np.min(res)
    norm_res = (res - min_res) / max_res-min_res
    return np.mean(norm_res)


def normalization(res):
    max_res = (res.max()).max()
    min_res = (res.min()).min()
    norm_res = (res - min_res) / max_res-min_res
    
    sum_res = norm_res.sum().sum()
    sum_res = sum_res/2
    
    n = len(res)
    count = (n * (n-1))/2
    return sum_res/count



def standardization(Data):
    mu = np.mean(Data, axis=0)
    sigma = np.std(Data, axis=0)
    
    return (Data - mu) / sigma



def calPYh_3(Data, cur_SF):
    sub_SFdata = Data.loc[:, cur_SF]
    sub_feature_std = sub_SFdata.std()
    return normalization_arr(list(sub_feature_std))


def cal_L2(Data, cur_SF):
    sub_SFdata = Data.loc[:, cur_SF]
    
    dis_df = pd.DataFrame(distance_matrix(sub_SFdata.values, sub_SFdata.values), index=sub_SFdata.index,columns=sub_SFdata.index)
    l2 = normalization(dis_df)
    return l2
    


def calFitness(Data, XcrossVec, alpha):
    cur_SF, cur_NSF = updateSet(XcrossVec)
    
    if len(cur_SF)==0:
        return -9999, [], cur_NSF

    phy3_val = calPYh_3(Data, cur_SF)
    
    l2_val = cal_L2(Data, cur_SF)
    penalty_len = len(cur_SF)/(len(cur_SF)+len(cur_NSF))

    # print(phy3_val, l2_val, alpha * penalty_len)
    fitness = phy3_val * (l2_val + alpha**2 * penalty_len)
    return fitness, cur_SF, cur_NSF
    