#coding:UTF-8
from numpy import *
from FitnessFun import calFitness
import random as rd
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize


# DE/current to best/1
def Mutation_cur2best_1(XTemp, Gbest, F):
    m, n = shape(XTemp) # m：row，n：column
    XMutationTmp = zeros((m, n))
    for i in range(m):
        r1 = 0
        r2 = 0
        while r1 == i or r2 == i or r1 == r2: # where i != r_1 != r_2 != r_3
            r1 = rd.randint(0, m - 1)
            r2 = rd.randint(0, m - 1)
        
        for j in range(n): # n: column
            XMutationTmp[i, j] = XTemp[i, j] + F * (Gbest[j] - XTemp[i, j]) + F * (XTemp[r1, j] - XTemp[r2, j]) 
        
    return XMutationTmp
    
    

# DE/current to rand/1
def Mutation_cur2rand_1(XTemp, Lbest, F):
    m, n = shape(XTemp) # m：row，n：column
    XMutationTmp = zeros((m, n))
    for i in range(m):
        r1 = 0
        r2 = 0
        while r1 == i or r2 == i or r1 == r2: # where i != r_1 != r_2 != r_3
            r1 = rd.randint(0, m - 1)
            r2 = rd.randint(0, m - 1)
        
        for j in range(n): # n: column
            XMutationTmp[i, j] = XTemp[i, j] + rd.random() * (XTemp[r1, j] - XTemp[i, j]) + F * (Lbest[j] - XTemp[r2, j]) 
        
    return XMutationTmp


# DE/rand/3
def Mutation_rand_3(XTemp, Lbest, F):
    m, n = shape(XTemp) # m：row，n：column
    XMutationTmp = zeros((m, n))
    for i in range(m):
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 0
        r5 = 0
        r6 = 0
        while r1 == i or r2 == i or r3 == i or r4 == i or r5 == i or r6 == i \
                or r1 == r2 or r1 == r3 or r1 == r4 or r1 == r5 or r1 == r6 \
                or r2 == r3 or r2 == r4 or r2 == r5 or r2 == r6 \
                or r3 == r4 or r3 == r5 or r3 == r6 \
                or r4 == r5 or r4 == r6 \
                or r5 == r6:
            r1 = rd.randint(0, m - 1)
            r2 = rd.randint(0, m - 1)
            r3 = rd.randint(0, m - 1)
            r4 = rd.randint(0, m - 1)
            r5 = rd.randint(0, m - 1)
            r6 = rd.randint(0, m - 1)
        
        for j in range(n): # n: column
            XMutationTmp[i, j] = XTemp[r1, j] + F * (Lbest[j] - XTemp[r2, j] + XTemp[r3, j] - XTemp[r4, j] + XTemp[r5, j] - XTemp[r6, j]) # v_{i,g} = v_{r_1,g} + F * (x_{r_2,g} - x_{r_3,g})
        
    return XMutationTmp


# DE/best/1
def Mutation_best_1(XTemp, Gbest, F):
    m, n = shape(XTemp) # m：row，n：column
    XMutationTmp = zeros((m, n))
    for i in range(m):
        r1 = 0
        r2 = 0
        while r1 == i or r2 == i or r1 == r2: # where i != r_1 != r_2 != r_3
            r1 = rd.randint(0, m - 1)
            r2 = rd.randint(0, m - 1)
        
        for j in range(n): # n: column
            XMutationTmp[i, j] = Gbest[j] + F * (XTemp[r1, j] - XTemp[r2, j])
        
    return XMutationTmp


# DE/rand to best/1
def Mutation_rand2best_1(XTemp, Gbest, F):
    m, n = shape(XTemp) # m：row，n：column
    XMutationTmp = zeros((m, n))
    for i in range(m):
        r1 = 0
        r2 = 0
        r3 = 0
        while r1 == i or r2 == i or r3 == i or r1 == r2 or r1 == r3 or r2 == r3: # where i != r_1 != r_2 != r_3
            r1 = rd.randint(0, m - 1)
            r2 = rd.randint(0, m - 1)
            r3 = rd.randint(0, m - 1)
        
        for j in range(n): # n: column
            XMutationTmp[i, j] = XTemp[r1, j] + F * (Gbest[j] - XTemp[i, j]) + F * (XTemp[r2, j] - XTemp[r3, j]) 
        
    return XMutationTmp
    

# DE/rand/2
def Mutation_rand_2(XTemp, Lbest, F):
    m, n = shape(XTemp) # m：row，n：column
    XMutationTmp = zeros((m, n))
    for i in range(m):
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 0
        while r1 == i or r2 == i or r3 == i or r4 == i \
                or r1 == r2 or r1 == r3 or r1 == r4 \
                or r2 == r3 or r2 == r4 \
                or r3 == r4:
            r1 = rd.randint(0, m - 1)
            r2 = rd.randint(0, m - 1)
            r3 = rd.randint(0, m - 1)
            r4 = rd.randint(0, m - 1)

        for j in range(n): # n: column
            XMutationTmp[i, j] = XTemp[r1, j] + F * (Lbest[j] - XTemp[r2, j] + XTemp[r3, j] - XTemp[r4, j]) # v_{i,g} = v_{r_1,g} + F * (x_{r_2,g} - x_{r_3,g})
        
    return XMutationTmp


# DE/best/2
def Mutation_best_2(XTemp, Gbest, F):
    m, n = shape(XTemp) # m：row，n：column
    XMutationTmp = zeros((m, n))
    for i in range(m):
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 0
        while r1 == i or r2 == i or r3 == i or r4 == i \
                or r1 == r2 or r1 == r3 or r1 == r4 \
                or r2 == r3 or r2 == r4 \
                or r3 == r4:
            r1 = rd.randint(0, m - 1)
            r2 = rd.randint(0, m - 1)
            r3 = rd.randint(0, m - 1)
            r4 = rd.randint(0, m - 1)
        
        for j in range(n): # n: column
            XMutationTmp[i, j] = Gbest[j] + F * (XTemp[r1, j] - XTemp[r2, j]) + F * (XTemp[r3, j] - XTemp[r4, j])
        
    return XMutationTmp


# DE/best/3
def Mutation_best_3(XTemp, Gbest, F):
    m, n = shape(XTemp) # m：row，n：column
    XMutationTmp = zeros((m, n))
    for i in range(m):
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 0
        r5 = 0
        r6 = 0
        while r1 == i or r2 == i or r3 == i or r4 == i or r5 == i or r6 == i \
                or r1 == r2 or r1 == r3 or r1 == r4 or r1 == r5 or r1 == r6 \
                or r2 == r3 or r2 == r4 or r2 == r5 or r2 == r6 \
                or r3 == r4 or r3 == r5 or r3 == r6 \
                or r4 == r5 or r4 == r6 \
                or r5 == r6:
            r1 = rd.randint(0, m - 1)
            r2 = rd.randint(0, m - 1)
            r3 = rd.randint(0, m - 1)
            r4 = rd.randint(0, m - 1)
            r5 = rd.randint(0, m - 1)
            r6 = rd.randint(0, m - 1)
        
        for j in range(n): # n: column
            XMutationTmp[i, j] = Gbest[j] + F * (XTemp[r1, j] - XTemp[r2, j] + XTemp[r3, j] - XTemp[r4, j] + XTemp[r5, j] - XTemp[r6, j]) # v_{i,g} = v_{r_1,g} + F * (x_{r_2,g} - x_{r_3,g})

    return XMutationTmp


def corr_mutation(Mutaion_list, choice_index, threshold_sim):
    choice_mutation = []
    Mutation_id = random.choice(choice_index)
    Mutation_rd = Mutaion_list[Mutation_id]
    choice_mutation.append(Mutation_rd)
    for per_id, perMutation in enumerate(Mutaion_list):
        if per_id == Mutation_id:
            continue

        norm1 = norm(Mutation_rd, axis=-1).reshape(Mutation_rd.shape[0], 1)
        norm2 = norm(perMutation, axis=-1).reshape(1, perMutation.shape[0])
        end_norm = np.dot(norm1, norm2)
        cos = np.dot(Mutation_rd, perMutation.T) / end_norm
        similarity = 0.5 * cos + 0.5

        if mean(similarity) <= threshold_sim:
            choice_mutation.append(perMutation)

    if len(choice_mutation) == 1:
        snd_choice_id = random.choice(choice_index)
        while snd_choice_id == Mutation_id:
            snd_choice_id = random.choice(choice_index)
        snd_choice = Mutaion_list[snd_choice_id]
        choice_mutation.append(snd_choice)
    return choice_mutation


# integrate whole mutation operator
def WeightedSum_mutation(XTemp, Gbest, Lbest, F, g, gamma, threshold_sim):

    # DE/current to best/1
    X_cur2best_1 = Mutation_cur2best_1(XTemp, Gbest, F)  # 2.

    # DE/current to rand/1
    X_cur2rand_1 = Mutation_cur2rand_1(XTemp, Lbest, F)  # 4.

    # DE/rand/3
    X_rand_3 = Mutation_rand_3(XTemp, Lbest, F)   # 8.

    # DE/best/1
    X_best_1 = Mutation_best_1(XTemp, Gbest, F)  # 1.

    # DE/rand to best/1
    X_rand2best_1 = Mutation_rand2best_1(XTemp, Gbest, F)  # 3.

    # DE/rand/2
    X_rand_2 = Mutation_rand_2(XTemp, Lbest, F)  # 6.

    # DE/best/2
    X_best_2 = Mutation_best_2(XTemp, Gbest, F)  # 5.

    # DE/best/3
    X_best_3 = Mutation_best_3(XTemp, Gbest, F)   # 7.

    Mutation_list = [X_best_1, X_cur2best_1, X_rand2best_1, X_cur2rand_1, X_best_2, X_rand_2, X_best_3, X_rand_3]
    choice_index = [0, 1, 2, 3, 4, 5, 6, 7]
    choice_mutation = corr_mutation(Mutation_list, choice_index, threshold_sim)

    XMutation = gamma**g * choice_mutation[0]
    i = 1
    for perChoice_mt in choice_mutation[1:]:
        XMutation += gamma**(g+i) * perChoice_mt
        i += 1

    return XMutation, len(choice_mutation)


def Mutation_std(Data):
    feature_std = Data.std()
    return mean(feature_std)


def crossover(XTemp, XMutationTmp, CR, old_Mut_matrix):
    m, n = shape(XTemp) # m: row, n: column
    XCorssOverTmp = zeros((m,n))
    new_Mut_matrix = Mutation_std(XMutationTmp)
    
    if new_Mut_matrix > old_Mut_matrix:
        old_Mut_matrix = new_Mut_matrix
        for i in range(m):
            jrand = rd.randint(0, n-1)
            for j in range(n): # column
                if (rd.random() <= 0.95) or (jrand == j):
                    XCorssOverTmp[i,j] = XMutationTmp[i,j]  # u^j_{i,g} = v^j_{i,g}
                else:
                    XCorssOverTmp[i,j] = XTemp[i,j]  # u^j_{i,g} = x^j_{i,g}
    
    else:
        for i in range(m):
            jrand = rd.randint(0, n-1)
            for j in range(n): # column
                if (rd.random() <= CR) or (jrand == j):
                    XCorssOverTmp[i,j] = XMutationTmp[i,j]  # u^j_{i,g} = v^j_{i,g}
                else:
                    XCorssOverTmp[i,j] = XTemp[i,j]  # u^j_{i,g} = x^j_{i,g}
    return XCorssOverTmp, old_Mut_matrix


def boundary_fun(XcrossVector, ub, lb):
    m, n = shape(XcrossVector) # m: row, n: column
    for i in range(m):
        Cross_vector = XcrossVector[i]
        for j in range(n): # column
            cur_ele = Cross_vector[j]
            if cur_ele > ub:
                XcrossVector[i][j] = ub
            elif cur_ele < lb:
                XcrossVector[i][j] = lb
    
    return XcrossVector


def selection(XTemp, XCorssOverTmp, fitnessVal, data, SF_list, NSF_list, alpha):

    m,n = shape(XTemp) # m: row, n: column
    fitnessCrossOverVal = zeros((m,1))
    for i in range(m):
         
        fitnessCrossOverVal[i,0], cur_SF, cur_NSF = calFitness(data, XCorssOverTmp[i], alpha)
        if (fitnessCrossOverVal[i,0] > fitnessVal[i,0]):
            for j in range(n):
                XTemp[i,j] = XCorssOverTmp[i,j]
            fitnessVal[i,0] = fitnessCrossOverVal[i,0]
            SF_list[i] = cur_SF
            NSF_list[i] = cur_NSF
    
    return XTemp, fitnessVal, SF_list, NSF_list


# global best
def saveBest(fitnessVal, XTmp, SF_list, NSF_list):
    m = shape(fitnessVal)[0] # row
    
    tmp = 0
    for i in range(1,m): # row iter
        if (fitnessVal[tmp] < fitnessVal[i]):
            tmp = i
    return XTmp[tmp], fitnessVal[tmp][0], SF_list[tmp], NSF_list[tmp]


# local best
def saveLBest(XCross, data, alpha):
    
    m,n = shape(XCross) # m: row, n: column
    fitnessCrossOverVal = zeros((m,1))
    for i in range(m):
         
        fitnessCrossOverVal[i,0], _, _ = calFitness(data, XCross[i], alpha)
    

    tmp = 0
    for i in range(1,m): # row iter
        if (fitnessCrossOverVal[tmp] < fitnessCrossOverVal[i]):
            tmp = i
    return XCross[tmp], fitnessCrossOverVal[tmp][0]



def IMoDE_model(NP, size, xMin, xMax, F, CR, generation, data, gamma, threshold_sim, alpha):

    best_fitness_list = []
    best_solution_list = []
    best_SF = []
    best_NSF = []


    XTemp = zeros((NP, size))
    for i in range(NP):
        for j in range(size):
            XTemp[i, j] = xMin + rd.random()*(xMax - xMin)  # x_{i,1} = rand(0, 1) * (b_{j,U} - b_{j,L}) + b_{j,L}
    old_mut_matrixs = Mutation_std(XTemp)
    

    fitnessVal = zeros((NP, 1))
    SF_list = []
    NSF_list = []  
    for i in range(NP):
        fitnessVal[i, 0], cur_SF, cur_NSF = calFitness(data, XTemp[i], alpha)
        SF_list.append(cur_SF)
        NSF_list.append(cur_NSF)


    Gbest, _, _, _ = saveBest(fitnessVal, XTemp, SF_list, NSF_list)
    Lbest = Gbest

    gen = 0  # generation
    choice_lenlist = []
    while gen <= generation:
        print('\nthe current generation', gen+1)

        XMutationTmp, len_choice = WeightedSum_mutation(XTemp, Gbest, Lbest, F, gen, gamma, threshold_sim)
        choice_lenlist.append(len_choice)

    
        XCorssOverTmp, old_mut_matrixs = crossover(XTemp, XMutationTmp, CR, old_mut_matrixs)


        XCorssOverTmp = boundary_fun(XCorssOverTmp, xMax, xMin)

        Lbest, _ = saveLBest(XCorssOverTmp, data, alpha)

        XTemp, fitnessVal, SF_list, NSF_list = \
            selection(XTemp, XCorssOverTmp, fitnessVal, data, SF_list, NSF_list, alpha)
        

        best_solution, best_fitness, best_SF_val, best_NSF_val = saveBest(fitnessVal, XTemp, SF_list, NSF_list)
        print('best solution: ')
        print(best_solution)
        print('best fitness: ')
        print(best_fitness)
        print('SF: ', best_SF_val)

        Gbest = best_solution  # update Gbest
        best_solution_list.append(best_solution)
        best_fitness_list.append(best_fitness)
        best_SF.append(best_SF_val)
        best_NSF.append(best_NSF_val)
        gen += 1

    return best_solution_list, best_fitness_list, best_SF, best_NSF, choice_lenlist
