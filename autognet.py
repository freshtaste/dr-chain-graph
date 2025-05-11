import numpy as np
from sklearn.linear_model import LogisticRegression
from agcEffect import agc_effect

def prepare_features_outcome_model(Y, A, L, adj_matrix):
    N = adj_matrix.shape[0]
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]
    X_list = []
    for i in range(N):
        A_i = A[i]
        A_sum = np.sum(A[neighbors[i]])
        L_i = L[i]
        L_sum = np.sum(L[neighbors[i]], axis=0) if len(neighbors[i]) > 0 else np.zeros(3)
        Y_sum = np.sum(Y[neighbors[i]]) if len(neighbors[i]) > 0 else 0
        #X_i = [A_i, A_sum] + L_i.tolist() + L_sum.tolist() + [Y_sum]
        X_i = [A_i, A_sum, L_i[0], L_sum[0], L_i[1], L_sum[1], L_i[2], L_sum[2], Y_sum]
        X_list.append(X_i)
    return np.array(X_list)

def prepare_features_L_k(L, k, adj_matrix):
    N = L.shape[0]
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]
    X_list, y_list = [], []
    for i in range(N):
        L_i = L[i]
        L_neighbors = L[neighbors[i]] if len(neighbors[i]) > 0 else np.zeros((0, 3))
        L_sum = np.sum(L_neighbors, axis=0) if L_neighbors.size else np.zeros(3)
        X = []
        for j in range(3):
            if j != k:
                X.append(L_i[j])
        X.extend(L_sum.tolist())
        X_list.append(X)
        y_list.append(L_i[k])
    return np.array(X_list), np.array(y_list)

def fit_autog_models(Y, A, L, adj_matrix):
    L_models = []
    for k in range(3):
        X_Lk, y_Lk = prepare_features_L_k(L, k, adj_matrix)
        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
        model.fit(X_Lk, y_Lk)
        L_models.append(model)
    X_Y = prepare_features_outcome_model(Y, A, L, adj_matrix)
    Y_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
    Y_model.fit(X_Y, Y)
    return {
        'L_models': L_models,
        'Y_model': Y_model
    }

def extract_parameters_from_autog_models(models, adj_matrix):
    """
    Extract tau, rho, nu, and beta from fitted autognet logistic models for use in agc_effect.
    Adjusted to include intercepts from model.intercept_ explicitly.
    """
    L_models = models['L_models']
    Y_model = models['Y_model']
    ncov = 3  # dimension of L

    # Extract tau, rho, nu
    tau = np.zeros(ncov)
    rho = np.zeros((ncov, ncov))
    nu = np.zeros((ncov, ncov))
    
    coef1 = L_models[0].coef_.flatten()
    intercept1 = L_models[0].intercept_[0]
    tau[0] = intercept1  # use intercept for tau
    rho[0, 1], rho[0, 2], nu[0, 0], nu[0, 1], nu[0, 2] = coef1[0], coef1[1], coef1[2], coef1[3], coef1[4]
    
    coef2 = L_models[1].coef_.flatten()
    intercept2 = L_models[1].intercept_[0]
    tau[1] = intercept2  # use intercept for tau
    rho[1, 0], rho[1, 2], nu[1, 0], nu[1, 1], nu[1, 2] = coef1[0], coef1[1], coef1[2], coef1[3], coef1[4]
    
    coef3 = L_models[2].coef_.flatten()
    intercept3 = L_models[2].intercept_[0]
    tau[2] = intercept3  # use intercept for tau
    rho[2, 0], rho[2, 1], nu[2, 0], nu[2, 1], nu[2, 2] = coef1[0], coef1[1], coef1[2], coef1[3], coef1[4]

    # Extract beta (intercept + coefficients)
    beta = np.concatenate([Y_model.intercept_, Y_model.coef_.flatten()])
    
    return tau, rho, nu, beta

def evaluate_autognet_via_agc_effect(adj_matrix, Y, A, L, treatment_allocation=0.5, R=10, burnin=5, seed=0):
    """
    Fit autognet models and evaluate causal effects using agc_effect.
    """
    models = fit_autog_models(Y, A, L, adj_matrix)
    tau, rho, nu, beta = extract_parameters_from_autog_models(models, adj_matrix)
    
    # print("tau:", tau)
    # print("rho:", rho)
    # print("nu:", nu)
    # print("beta:", beta)
    
    ret = agc_effect(
        adj_matrix=adj_matrix,
        tau=tau,
        rho=rho,
        nu=nu,
        beta=beta,
        treatment_allocation=treatment_allocation,
        R=R,
        burnin_R=burnin,
        seed=seed
    )

    print("psi_zero:", ret['psi_zero'])
    
    return ret
