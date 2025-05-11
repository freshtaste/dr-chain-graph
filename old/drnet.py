import numpy as np
from sklearn.linear_model import LogisticRegression

def expit(x):
    return 1 / (1 + np.exp(-x))

def get_neighbor_summary(X, adj_matrix):
    return adj_matrix @ X

def fit_logistic_model(X, y):
    model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=10000)
    model.fit(X, y)
    return model

def build_design_matrix_Y(A, L, Y, adj_matrix):
    A_nb = get_neighbor_summary(A.reshape(-1, 1), adj_matrix).flatten()
    L_nb = get_neighbor_summary(L, adj_matrix)
    Y_nb = get_neighbor_summary(Y.reshape(-1, 1), adj_matrix).flatten()
    return np.column_stack([
        A,
        A_nb,
        L[:, 0], L_nb[:, 0],
        L[:, 1], L_nb[:, 1],
        L[:, 2], L_nb[:, 2],
        Y_nb
    ])

def build_design_matrix_A(L, A, adj_matrix):
    L_nb = get_neighbor_summary(L, adj_matrix)
    A_nb = get_neighbor_summary(A.reshape(-1, 1), adj_matrix).flatten()
    return np.column_stack([
        L[:, 0], L_nb[:, 0],
        L[:, 1], L_nb[:, 1],
        L[:, 2], L_nb[:, 2],
        A_nb
    ])

def compute_beta_probs(X_y, model_y):
    return model_y.predict_proba(X_y)[:, 1]


def compute_a_group_linpred(i, a_vector, A, L, gamma, adj_matrix):
    N = adj_matrix.shape[0]
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]
    group = np.concatenate(([i], neighbors[i]))
    aG = []
    for k in group:
        G_k = (
                gamma[0]
                + gamma[1] * L[k, 0] + gamma[2] * np.sum(L[neighbors[k], 0])
                + gamma[3] * L[k, 1] + gamma[4] * np.sum(L[neighbors[k], 1])
                + gamma[5] * L[k, 2] + gamma[6] * np.sum(L[neighbors[k], 2])
            )
        aG.append(a_vector[k] * G_k)
    aG = np.sum(aG)
    
    aA = []
    for k in list(group):
        for j in list(group):
            if k < j and adj_matrix[k, j] == 1:
                aA.append(a_vector[k] * a_vector[j] * gamma[7])
                
    for k in list(group):
        for j in range(N):
            if j not in list(group) and adj_matrix[k, j] == 1:
                aA.append(a_vector[k] * A[j] * gamma[7])
    aA = np.sum(aA)
    
    return aG + aA

from itertools import product

def generate_binary_vectors(m):
    return list(product([0, 1], repeat=m))

def compute_pi_probs(i, a_vector, A, L, adj_matrix, model_a):
    gamma = np.concatenate([model_a.intercept_, model_a.coef_.flatten()])
    numerator = np.exp(compute_a_group_linpred(i, a_vector, A, L, gamma, adj_matrix))
    
    neighbours = np.where(adj_matrix[i])[0]
    a_groups = generate_binary_vectors(len(neighbours)+1)
    print(len(a_groups))
    
    denominator = []
    for a_group in a_groups:
        a_vector_temp = a_vector.copy()
        a_vector_temp[neighbours] = a_group[1:]
        a_vector_temp[i] = a_group[0]
        denominator.append(np.exp(compute_a_group_linpred(i, a_vector_temp, A, L, gamma, adj_matrix)))
    denominator = np.sum(denominator)
    
    pi = numerator / denominator
    return pi
    

def doubly_robust_IF_i(i, a_vector, A, L, Y, model_y, model_a, adj_matrix):
    neighbors = np.where(adj_matrix[i])[0]
    group = np.concatenate(([i], neighbors))
    a_group = tuple(a_vector[group])

    A_temp = A.copy()
    A_temp[group] = a_vector[group]
    A_nb = get_neighbor_summary(A_temp.reshape(-1, 1), adj_matrix).flatten()
    L_nb = get_neighbor_summary(L, adj_matrix)
    Y_nb = get_neighbor_summary(Y.reshape(-1, 1), adj_matrix).flatten()

    X_y = np.array([[A_temp[i],
                     A_nb[i],
                     L[i, 0], L_nb[i, 0],
                     L[i, 1], L_nb[i, 1],
                     L[i, 2], L_nb[i, 2],
                     Y_nb[i]]])
    beta_hat = model_y.predict_proba(X_y)[0, 1]
    
    if tuple(A[group]) != a_group:
        return beta_hat
    else:
        pi = compute_pi_probs(i, a_vector, A, L, adj_matrix, model_a)
        return 1 / pi * (Y[i] - beta_hat) + beta_hat

from tqdm import tqdm

def doubly_robust_IF(a_vector, A, L, Y, adj_matrix):
    # fit models
    X_y = build_design_matrix_Y(A, L, Y, adj_matrix)
    model_y = fit_logistic_model(X_y, Y)
    X_a = build_design_matrix_A(L, A, adj_matrix)
    model_a = fit_logistic_model(X_a, A)
    
    # compute influence function
    N = adj_matrix.shape[0]
    psi = np.zeros(N)
    for i in tqdm(range(N)):
        psi[i] = doubly_robust_IF_i(i, a_vector, A, L, Y, model_y, model_a, adj_matrix)
    return psi

def doubly_robust(A, L, Y, adj_matrix, treatment_allocation):
    # fit models
    X_y = build_design_matrix_Y(A, L, Y, adj_matrix)
    model_y = fit_logistic_model(X_y, Y)
    X_a = build_design_matrix_A(L, A, adj_matrix)
    model_a = fit_logistic_model(X_a, A)
    
    # compute influence function
    N = adj_matrix.shape[0]
    psi_gamma, psi_zero, psi_1_gamma, psi_0_gamma = [], [], [], []
    for i in tqdm(range(N)):
        a_vector = np.random.binomial(1, treatment_allocation, size=N)
        psi_gamma.append(doubly_robust_IF_i(i, a_vector, A, L, Y, model_y, model_a, adj_matrix))
        
        a_vector = np.zeros(N)
        psi_zero.append(doubly_robust_IF_i(i, a_vector, A, L, Y, model_y, model_a, adj_matrix))
        
        a_vector[i] = 1
        psi_1_gamma.append(doubly_robust_IF_i(i, a_vector, A, L, Y, model_y, model_a, adj_matrix))
        
        a_vector[i] = 0
        psi_0_gamma.append(doubly_robust_IF_i(i, a_vector, A, L, Y, model_y, model_a, adj_matrix))
    
    avg_psi_gamma = np.mean(psi_gamma)
    direct_effect = np.mean(psi_1_gamma) - np.mean(psi_0_gamma)
    spillover_effect = np.mean(psi_0_gamma) - np.mean(psi_zero)
    return {
        'avg_psi_gamma': avg_psi_gamma,
        'direct_effect': direct_effect,
        'spillover_effect': spillover_effect
    }