from read_rawData import get_rawData
from IMoDE import IMoDE_model
from classifier import classifiers_func
from utils import process_label, fill_data, str2int, standa_raw, record_res
from evaluate import Metrics
from plot_fig import fig, choiceM_fig
from saveRes import list2csv
from sklearn.model_selection import KFold
import numpy as np


# set param
name_ = 'hypothyroid'
path = 'Data/hypothyroid.csv'
NP = 100 # size of the population, row.
xMin = 0 # b_{j,L}
xMax = 1 # b_{j,U}
F = 0.8 # the scaling factor
CR = 0.8 # the crossover rate
gamma = 0.95
generation = 100
threshold_sim = 0.75
alpha = 0.75
IsID=False
Isheader = 0
kfold_num = 5
kfold_shuffle = True
kfold_seed = 75


# processed the data
data = get_rawData(path, Isheader, IsID)
new_columnName = [i for i in range(len(list(data.columns)))]
data.columns = new_columnName
data = process_label(data)
data = fill_data(data)
data = str2int(data)
data = standa_raw(data)

ISmcc = True if len(data['label'].unique())==2 else False


# kfolds
kf = KFold(n_splits=kfold_num, shuffle=kfold_shuffle, random_state=kfold_seed)
cur_k = 1
Metrics_dict = {}
chosen_SF_list = []
for train_index, test_index in kf.split(data):
    # print('train_index', train_index, 'test_index', test_index)
    X_train, Y_train = data.iloc[train_index, 0:-1], data.iloc[train_index, -1]
    X_test, Y_test = data.iloc[test_index, 0:-1], data.iloc[test_index, -1]
    X_train = X_train.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)



    size = len(list(X_train.columns))  # dimention of the per size, column
    best_solution_list, best_fitness_list, best_SF, best_NSF, choice_lenlist = IMoDE_model(NP, size, xMin, xMax, F, CR, generation, X_train, gamma, threshold_sim, alpha)
    
    
    # 特征子集训练模型
    classifiers, X_test, Y_test = classifiers_func(X_train, Y_train, X_test, Y_test, best_SF[-1])
    
    
    # 评估模型
    metrics = Metrics(classifiers, X_test, Y_test, ISmcc)
    metrics['length'] = len(best_SF[-1])
    Metrics_dict[cur_k] = metrics
    chosen_SF_list.append(len(best_SF[-1]))

    cur_k += 1

record_res(Metrics_dict, len(list(data.iloc[:, :-1].columns)), ISmcc, name_)