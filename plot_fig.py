import matplotlib.pyplot as plt


def fig(y_list, name_, cur_k):
    x_list = [i + 1 for i in range(len(y_list))]

    plt.plot(x_list, y_list)
    plt.savefig('Res/Gbest_fitness_' + name_ + '_' + str(cur_k) + '.jpg')


def choiceM_fig(choice_lenlist, name_, cur_k):
    x_list = [i + 1 for i in range(len(choice_lenlist))]

    plt.plot(x_list, choice_lenlist)
    plt.savefig('Res/choice_mutation_len_' + name_ + '_' + str(cur_k) + '.jpg')