import numpy as np
from sklearn.linear_model import LogisticRegression

def expit(x):
    return 1 / (1 + np.exp(-x))

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
        X_i = [A_i, A_sum] + L_i.tolist() + L_sum.tolist() + [Y_sum]
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
        model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=10000)
        model.fit(X_Lk, y_Lk)
        L_models.append(model)
    X_Y = prepare_features_outcome_model(Y, A, L, adj_matrix)
    Y_model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=10000)
    Y_model.fit(X_Y, Y)
    return {
        'L_models': L_models,
        'Y_model': Y_model
    }

def sample_autog_gibbs(adj_matrix, models, R=10, burnin=5, seed=0, treatment_allocation=0.5,
                       Atype='all'):
    np.random.seed(seed)
    N = adj_matrix.shape[0]
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]
    L = np.random.binomial(1, 0.5, size=(N, 3))
    Y = np.random.binomial(1, 0.5, size=N)
    L_chain = []
    Y_chain = []
    L_models = models['L_models']
    Y_model = models['Y_model']

    def feature_Lk(i, k):
        X = []
        for j in range(3):
            if j != k:
                X.append(L[i, j])
        L_sum = np.sum(L[neighbors[i]], axis=0) if len(neighbors[i]) > 0 else np.zeros(3)
        X.extend(L_sum.tolist())
        return np.array(X).reshape(1, -1)

    def feature_Y(i, A):
        A_i = A[i]
        A_sum = np.sum(A[neighbors[i]]) if len(neighbors[i]) > 0 else 0
        L_i = L[i]
        L_sum = np.sum(L[neighbors[i]], axis=0) if len(neighbors[i]) > 0 else np.zeros(3)
        Y_sum = np.sum(Y[neighbors[i]]) if len(neighbors[i]) > 0 else 0
        X = [A_i, A_sum] + L_i.tolist() + L_sum.tolist() + [Y_sum]
        return np.array(X).reshape(1, -1)

    for m in range(R + burnin):
        for i in range(N):
            if Atype == 'all':
                A = np.random.binomial(1, treatment_allocation, size=N)
            elif Atype == 'ind_treat_1':
                A = np.random.binomial(1, treatment_allocation, size=N)
                A[i] = 1
            elif Atype == 'ind_treat_0':
                A = np.random.binomial(1, treatment_allocation, size=N)
                A[i] = 0
            elif Atype == 'all_0':
                A = np.zeros(N)
            else:
                raise ValueError("Invalid Atype. Choose from 'all', 'ind_treat_1', 'ind_treat_0', or 'all_0'.")
            X_Y_i = feature_Y(i, A)
            prob_Y = Y_model.predict_proba(X_Y_i)[0, 1]
            Y[i] = np.random.binomial(1, prob_Y)
            
            for k in range(3):
                X_Lk_i = feature_Lk(i, k)
                prob = L_models[k].predict_proba(X_Lk_i)[0, 1]
                L[i, k] = np.random.binomial(1, prob)
        if m >= burnin:
            L_chain.append(L.copy())
            Y_chain.append(Y.copy())

    return np.array(L_chain), np.array(Y_chain)

def evaluate_autog_effect(adj_matrix, Y, A, L, treatment_allocation=0.5, R=10, burnin=5, seed=0):
    models = fit_autog_models(Y, A, L, adj_matrix)
    
    psi_gamma, psi_zero, psi_1_gamma, psi_0_gamma = [], [], [], []
        
    # 1. Average outcome with treatment assigned at rate p
    L_chain, Y_chain = sample_autog_gibbs(adj_matrix, models, R, burnin, seed, treatment_allocation, Atype='all')
    for L, Y in zip(L_chain, Y_chain):
        psi_gamma.append(np.mean(Y))

    # 2. All control
    L_chain, Y_chain = sample_autog_gibbs(adj_matrix, models, R + burnin, burnin, seed, treatment_allocation, Atype='all_0')
    for L, Y in zip(L_chain, Y_chain):
        psi_zero.append(np.mean(Y))
    
    # 3. Individual treated, neighbors treated at rate p
    L_chain, Y_chain = sample_autog_gibbs(adj_matrix, models, R + burnin, burnin, seed, treatment_allocation, Atype='ind_treat_1')
    for L, Y in zip(L_chain, Y_chain):
        psi_1_gamma.append(np.mean(Y))
        
    # 4. Individual not treated, neighbors treated at rate p
    L_chain, Y_chain = sample_autog_gibbs(adj_matrix, models, R + burnin, burnin, seed, treatment_allocation, Atype='ind_treat_0')
    for L, Y in zip(L_chain, Y_chain):
        psi_0_gamma.append(np.mean(Y))
        
    # Compute effects
    avg_psi_gamma = np.mean(psi_gamma)
    direct_effect = np.mean(psi_1_gamma) - np.mean(psi_0_gamma)
    spillover_effect = np.mean(psi_0_gamma) - np.mean(psi_zero)
    
    print("psi_zero:", psi_zero)
    
    return {
        "average": avg_psi_gamma,
        "direct_effect": direct_effect,
        "spillover_effect": spillover_effect,
        "psi_gamma": np.mean(psi_gamma),
        "psi_1_gamma": np.mean(psi_1_gamma),
        "psi_0_gamma": np.mean(psi_0_gamma),
        "psi_zero": np.mean(psi_zero),
    }