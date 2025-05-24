import numpy as np
from sklearn.linear_model import LogisticRegression
from dgp import agcEffect

def prepare_features_outcome_model(Y, A, L, adj_matrix):
    N, D = L.shape
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]
    X_list = []
    for i in range(N):
        A_i = A[i]
        A_sum = np.sum(A[neighbors[i]])
        L_i = L[i]
        L_sum = np.sum(L[neighbors[i]], axis=0) if len(neighbors[i]) > 0 else np.zeros(D)
        Y_sum = np.sum(Y[neighbors[i]]) if len(neighbors[i]) > 0 else 0

        # Build feature vector: A_i, A_sum, L_i and L_sum interleaved, then Y_sum
        X_i = [A_i, A_sum]
        for d in range(D):
            X_i.extend([L_i[d], L_sum[d]])
        X_i.append(Y_sum)
        X_list.append(X_i)
    return np.array(X_list)


def prepare_features_L_k(L, k, adj_matrix):
    N, D = L.shape
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]
    X_list, y_list = [], []
    for i in range(N):
        L_i = L[i]
        L_neighbors = L[neighbors[i]] if len(neighbors[i]) > 0 else np.zeros((0, D))
        L_sum = np.sum(L_neighbors, axis=0) if L_neighbors.size else np.zeros(D)
        X = []
        for j in range(D):
            if j != k:
                X.append(L_i[j])
        X.extend(L_sum.tolist())
        X_list.append(X)
        y_list.append(L_i[k])
    return np.array(X_list), np.array(y_list)


def fit_autog_models(Y, A, L, adj_matrix):
    D = L.shape[1]
    L_models = []
    for k in range(D):
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
    D = len(L_models)

    # Initialize
    tau = np.zeros(D)
    rho = np.zeros((D, D))
    nu = np.zeros((D, D))

    for k in range(D):
        coef = L_models[k].coef_.flatten()
        intercept = L_models[k].intercept_[0]
        tau[k] = intercept
        idx = 0
        for j in range(D):
            if j != k:
                rho[k, j] = coef[idx]
                idx += 1
        for j in range(D):
            nu[k, j] = coef[idx]
            idx += 1

    # Extract beta (intercept + coefficients)
    beta = np.concatenate([Y_model.intercept_, Y_model.coef_.flatten()])
    
    return tau, rho, nu, beta


from dgp import sample_Y1

def evaluate_autognet_via_agc_effect(adj_matrix, Y, A, L, treatment_allocation=0.5, R=10, burnin=5, seed=0):
    """
    Fit autognet models and evaluate causal effects using agc_effect.
    """
    np.random.seed(seed)
    models = fit_autog_models(Y, A, L, adj_matrix)
    tau, rho, nu, beta = extract_parameters_from_autog_models(models, adj_matrix)

    ret = agcEffect(
        adj_matrix=adj_matrix,
        tau=tau,
        rho=rho,
        nu=nu,
        beta=beta,
        treatment_allocation=treatment_allocation,
        R=R,
        burnin_R=burnin,
        seed=seed,
        sample_Y_func=sample_Y1
    )
    
    return ret
