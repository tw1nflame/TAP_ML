import numpy as np
import networkx as nx


def leblanc_3(T_0, D, C, e=0.1):
    n = D.shape[1]
    stop = False
    G = nx.DiGraph(T_0)
    X_1 = get_flow_vector(G, n, D)
    iteration_num = 1

    while not stop:
        T_1 = T_0 * (1 + 0.15 * (X_1 / C)**4)
        T_1[np.isnan(T_1)] = 0
        G_1 = nx.DiGraph(T_1)

        Y_1 = get_flow_vector(G_1, n, D)
        C_calc = 1 / (C**4)
        C_calc[np.isinf(C_calc)] = 0

        h = 0.001
        l_values = np.arange(0, 1 + h, h)
        lambda_val = 0.5

        for i in range(len(l_values) - 1):
            df_i = np.sum(T_0 * (Y_1 - X_1) + 0.15 * T_0 * C_calc *
                          ((X_1 + l_values[i] * (Y_1 - X_1))**4) * (Y_1 - X_1))
            df_iplus1 = np.sum(T_0 * (Y_1 - X_1) + 0.15 * T_0 * C_calc *
                               ((X_1 + l_values[i + 1] * (Y_1 - X_1))**4) * (Y_1 - X_1))

            if np.sign(df_i) != np.sign(df_iplus1):
                lambda_val = h * i + h / 2
                break

        X_2 = X_1 + lambda_val * (Y_1 - X_1)
        dif_val = (X_2 - X_1) / X_1
        dif_val[np.isnan(dif_val)] = 0
        dif_val[np.isinf(dif_val)] = 0

        if np.max(np.abs(dif_val)) < e:
            stop = True
        else:
            X_1 = X_2
            iteration_num += 1

    result_vector = X_1
    result_function_value = np.sum(T_0 * X_1 + T_0 * 0.03 * C_calc * (X_1**5))

    return result_vector, iteration_num, result_function_value


def get_flow_vector(G, n, D):
    X = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    path = nx.shortest_path(
                        G, source=i, target=j, weight='weight')
                    for k in range(len(path) - 1):
                        X[path[k], path[k + 1]] += D[i, j]
                except nx.NetworkXNoPath:
                    continue
    return X
