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

def compute_pi_probs(i, A, L, model_a, adj_matrix, R=2000, burnin=1000):
    N = adj_matrix.shape[0]
    neighbors = np.where(adj_matrix[i])[0]
    group = np.concatenate(([i], neighbors))
    K = len(group)

    A_sample = A.copy()
    counts = {}

    for it in range(R):
        for j in group:
            L_j = L[j].reshape(1, -1)
            L_nb_j = get_neighbor_summary(L, adj_matrix)[j].reshape(1, -1)
            A_nb_j = get_neighbor_summary(A_sample.reshape(-1, 1), adj_matrix)[j].reshape(1, -1)
            X_j = np.column_stack([L_j[:, 0], L_nb_j[:, 0],
                                   L_j[:, 1], L_nb_j[:, 1],
                                   L_j[:, 2], L_nb_j[:, 2],
                                   A_nb_j[:, 0]])
            p_j = model_a.predict_proba(X_j)[0, 1]
            A_sample[j] = np.random.binomial(1, p_j)

        if it >= burnin:
            key = tuple(A_sample[group])
            counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}
    return probs

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
        pi_dict = compute_pi_probs(i, A, L, model_a, adj_matrix)
        pi_hat = pi_dict.get(a_group, 1e-6)  # avoid zero probability
        return 1 / pi_hat * (Y[i] - beta_hat) + beta_hat

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
