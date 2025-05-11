import numpy as np
from sklearn.linear_model import LogisticRegression

def expit(x):
    return 1 / (1 + np.exp(-x))

def get_neighbor_summary(X, adj_matrix):
    return adj_matrix @ X

def fit_logistic_model(X, y):
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
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
    
def compute_beta_probs(X_y, model_y, Atype='ind_treat_1'):
    if Atype == 'ind_treat_1':
        X_y[:, 0] = 1
    elif Atype == 'ind_treat_0':
        X_y[:, 0] = 0
    elif Atype == 'all_0':
        X_y[:, 0] = 0
        X_y[:, 1] = 0
    elif Atype == 'all':
        pass
    else:
        raise ValueError("Invalid Atype specified.")
    return model_y.predict_proba(X_y)[:, 1]



def ocnet(A, L, Y, adj_matrix, treatment_allocation=0.7, num_rep=1000, seed=1):
    np.random.seed(seed)
    
    # fit models
    a_mat = np.random.binomial(1, treatment_allocation, size=(Y.shape[0], num_rep))
    X_y = build_design_matrix_Y(A, L, Y, adj_matrix)
    model_y = fit_logistic_model(X_y, Y)
    
    psi_gamma = []
    for i in range(num_rep):
        X_y_eval = build_design_matrix_Y(a_mat[:,i], L, Y, adj_matrix)
        beta_hat = compute_beta_probs(X_y_eval, model_y, Atype='all')
        psi = beta_hat 
        psi_gamma.append(psi.mean())
    
    psi_1_gamma = []
    for i in range(num_rep):
        X_y_eval = build_design_matrix_Y(a_mat[:,i], L, Y, adj_matrix)
        beta_hat = compute_beta_probs(X_y_eval, model_y, Atype='ind_treat_1')
        psi = beta_hat
        psi_1_gamma.append(psi.mean())
        
    psi_0_gamma = []
    for i in range(num_rep):
        X_y_eval = build_design_matrix_Y(a_mat[:,i], L, Y, adj_matrix)
        beta_hat = compute_beta_probs(X_y_eval, model_y, Atype='ind_treat_0')
        psi = beta_hat 
        psi_0_gamma.append(psi.mean())
    
    psi_zero = []
    a_mat = np.zeros((Y.shape[0],1))
    X_y_eval = build_design_matrix_Y(a_mat, L, Y, adj_matrix)
    beta_hat = compute_beta_probs(X_y_eval, model_y, Atype='all_0')
    psi = beta_hat 
    psi_zero.append(psi.mean())
    
    # Compute effects
    avg_psi_gamma = np.mean(psi_gamma)
    direct_effect = np.mean(psi_1_gamma) - np.mean(psi_0_gamma)
    spillover_effect = np.mean(psi_0_gamma) - np.mean(psi_zero)
    # print("psi_zero:", psi_zero)
    # print("beta_hat:", beta_hat.mean())
    # print("psi_0_gamma:", np.mean(psi_0_gamma))
    # print("psi_1_gamma:", np.mean(psi_1_gamma))
    # print("average:", np.mean(psi_gamma))
    # print("direct_effect:", direct_effect)
    # print("spillover_effect:", spillover_effect)
    
    return {
        "average": avg_psi_gamma,
        "direct_effect": direct_effect,
        "spillover_effect": spillover_effect,
        "psi_1_gamma": np.mean(psi_1_gamma),
        "psi_0_gamma": np.mean(psi_0_gamma),
        "psi_zero": np.mean(psi_zero),
    }