import  pandas as pd

def list2csv(best_solution_list, best_fitness_list, best_SF, best_NSF, name_, cur_k):

    best_sol_df = pd.DataFrame({})
    best_fitness_df = pd.DataFrame({})
    best_SF_df = pd.DataFrame({})
    best_NSF_df = pd.DataFrame({})

    best_sol_df['best solution'] = best_solution_list
    best_fitness_df['best fitness'] = best_fitness_list
    best_SF_df['best SF'] = best_SF
    best_NSF_df['best NSF'] = best_NSF

    best_sol_df.to_csv('Res/best_solution_' + name_ + '_' + str(cur_k) + '.csv', index=False)
    best_fitness_df.to_csv('Res/best_fitness_' + name_ + '_' + str(cur_k) + '.csv', index=False)
    best_SF_df.to_csv('Res/best_SF_' + name_ + '_' + str(cur_k) + '.csv', index=False)
    best_NSF_df.to_csv('Res/best_NSF_' + name_ + '_' + str(cur_k) +'.csv', index=False)


