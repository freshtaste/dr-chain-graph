import numpy as np
from sklearn.linear_model import LogisticRegression

def expit(x):
    return 1 / (1 + np.exp(-x))

def get_neighbor_summary(X, adj_matrix):
    return adj_matrix @ X

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

def get_2hop_neighbors(adj_matrix):
    """
    Returns for each node:
      - one_hop: list of direct neighbors
      - two_hop: list of neighbors-of-neighbors excluding direct neighbors and the node itself
    """
    N = adj_matrix.shape[0]
    one_hop = [np.where(adj_matrix[i])[0] for i in range(N)]
    two_hop = []
    for i in range(N):
        # gather neighbors of neighbors
        second = set()
        for j in one_hop[i]:
            second.update(np.where(adj_matrix[j])[0])
        # remove direct neighbors and self
        second.difference_update(one_hop[i])
        second.discard(i)
        two_hop.append(sorted(second))
    return one_hop, two_hop

def encode_pattern(pattern):
    """
    Encode pattern with length sensitivity by using tuple hashing.
    """
    return hash(tuple(pattern))

def prepare_features_propensity_model(A, L, adj_matrix):
    """
    Constructs feature matrix X_prop and label vector y_prop for multinomial logistic
    regression of joint treatment A_i and A_neighbors.
    Features: L_i, sum(L_one_hop), sum(L_two_hop), sum(A_two_hop)
    """
    N = adj_matrix.shape[0]
    one_hop, two_hop = get_2hop_neighbors(adj_matrix)
    # maximum pattern length = 1 + max degree
    max_deg = max(len(one_hop[i]) for i in range(N))
    max_len = 1 + max_deg

    X_list, y_list = [], []
    for i in range(N):
        # features
        Li = L[i]
        L1 = np.sum(L[one_hop[i]], axis=0) if len(one_hop[i]) > 0 else np.zeros(3)
        L2 = np.sum(L[two_hop[i]], axis=0) if len(two_hop[i]) > 0 else np.zeros(3)
        A2 = np.sum(A[two_hop[i]]) if len(two_hop[i]) > 0 else 0
        features = np.concatenate([Li, L1, L2, [A2]])
        X_list.append(features)

        # label: pattern of [A_i] + [A_j for j in one_hop[i]]
        pattern = [A[i], np.sum(A[one_hop[i]])]
        label = encode_pattern(pattern)
        y_list.append(label)

    X_prop = np.array(X_list)
    y_prop = np.array(y_list)
    return X_prop, y_prop

def estimate_models(Y, A, L, adj_matrix):
    """
    Estimate:
      - outcome_model: logistic regression for Y ~ features_from prepare_features_outcome_model
      - propensity_model: multinomial logistic regression for joint A_i and A_neighbors
    Returns fitted sklearn model objects and auxiliary info.
    """
    # Outcome model
    X_y = build_design_matrix_Y(A, L, Y, adj_matrix)
    outcome_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
    outcome_model.fit(X_y, Y)

    # Propensity model
    X_p, y_p = prepare_features_propensity_model(A, L, adj_matrix)

    # remove y_p with unfrequent labels
    unique_labels, counts = np.unique(y_p, return_counts=True)
    frequent_labels = unique_labels[counts > 20]
    mask = np.isin(y_p, frequent_labels)
    X_p = X_p[mask]
    y_p = y_p[mask]

    # fit multinomial logistic regression
    propensity_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
    propensity_model.fit(X_p, y_p)

    return {
        'outcome_model': outcome_model,
        'propensity_model': propensity_model
    }

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

def doubly_robust_em(A, L, Y, adj_matrix, treatment_allocation=0.7, num_rep=1000, seed=1, return_raw=False, psi_0_gamma_only=False):
    """
    Compute the doubly robust estimator for the average treatment effect.
    """
    N = adj_matrix.shape[0]
    models = estimate_models(Y, A, L, adj_matrix)
    outcome_model = models['outcome_model']
    propensity_model = models['propensity_model']
    X_prop, y_prop = prepare_features_propensity_model(A, L, adj_matrix)
    one_hop, two_hop = get_2hop_neighbors(adj_matrix)
    prop_model_pred = propensity_model.predict_proba(X_prop)

    # compute the influence function
    a_mat = np.random.binomial(1, treatment_allocation, size=(Y.shape[0], num_rep))
    if psi_0_gamma_only is False:
        psi_gamma = np.zeros((N, num_rep))
        psi_1_gamma = np.zeros((N, num_rep))
        psi_0_gamma = np.zeros((N, num_rep))
        for i in range(num_rep):
            # compute psi_gamma
            X_y_eval = build_design_matrix_Y(a_mat[:,i], L, Y, adj_matrix)
            beta_hat = compute_beta_probs(X_y_eval, outcome_model, Atype='all')
            prop_vec, I = np.zeros(N), np.zeros(N)
            for k in range(N):
                pattern = [a_mat[k,i], np.sum(a_mat[one_hop[k],i])]
                label = encode_pattern(pattern)
                if label not in propensity_model.classes_:
                    prop_vec[k] = 1e-4
                    I[k] = 0
                else:
                    label_idx = np.where(propensity_model.classes_ == label)[0][0]
                    prop_vec[k] = prop_model_pred[k,label_idx]
                    I[k] = 1 if a_mat[k,i] == A[k] and np.sum(a_mat[one_hop[k],i]) == np.sum(A[one_hop[k]]) else 0
            w = I / prop_vec
            w_norm = w/np.sum(w)*N if np.sum(w) > 0 else 0
            psi = beta_hat + w_norm * (Y - beta_hat)
            psi_gamma[:, i] = psi.copy()
            
            # compute psi_1_gamma
            beta_hat = compute_beta_probs(X_y_eval, outcome_model, Atype='ind_treat_1')
            prop_vec, I = np.zeros(N), np.zeros(N)
            for k in range(N):
                pattern = [1, np.sum(a_mat[one_hop[k],i])]
                label = encode_pattern(pattern)
                if label not in propensity_model.classes_:
                    prop_vec[k] = 1e-4
                    I[k] = 0
                else:
                    label_idx = np.where(propensity_model.classes_ == label)[0][0]
                    prop_vec[k] = prop_model_pred[k,label_idx]
                    I[k] = 1 if 1 == A[k] and np.sum(a_mat[one_hop[k],i]) == np.sum(A[one_hop[k]]) else 0
            w = I / prop_vec
            w_norm = w/np.sum(w)*N if np.sum(w) > 0 else 0
            psi = beta_hat + w_norm * (Y - beta_hat)
            psi_1_gamma[:, i] = psi.copy()
    else:
        psi_gamma = np.zeros((N, num_rep))
        psi_1_gamma = np.zeros((N, num_rep))

    # compute psi_0_gamma
    for i in range(num_rep):
        X_y_eval = build_design_matrix_Y(a_mat[:,i], L, Y, adj_matrix)
        beta_hat = compute_beta_probs(X_y_eval, outcome_model, Atype='ind_treat_0')
        prop_vec, I = np.zeros(N), np.zeros(N)
        for k in range(N):
            pattern = [0, np.sum(a_mat[one_hop[k],i])]
            label = encode_pattern(pattern)
            if label not in propensity_model.classes_:
                prop_vec[k] = 1e-4
                I[k] = 0
            else:
                label_idx = np.where(propensity_model.classes_ == label)[0][0]
                prop_vec[k] = prop_model_pred[k,label_idx]
                I[k] = 1 if 0 == A[k] and np.sum(a_mat[one_hop[k],i]) == np.sum(A[one_hop[k]]) else 0
        w = I / prop_vec
        w_norm = w/np.sum(w)*N if np.sum(w) > 0 else 0
        psi = beta_hat + w_norm * (Y - beta_hat)
        psi_0_gamma[:, i] = psi.copy()

    if psi_0_gamma_only is False:
        a_mat = np.zeros((Y.shape[0],1))
        X_y_eval = build_design_matrix_Y(a_mat, L, Y, adj_matrix)
        beta_hat = compute_beta_probs(X_y_eval, outcome_model, Atype='all_0')
        prop_vec, I = np.zeros(N), np.zeros(N)
        for k in range(N):
            pattern = [0, 0]
            label = encode_pattern(pattern)
            if label not in propensity_model.classes_:
                prop_vec[k] = 1e-4
                I[k] = 0
            else:
                label_idx = np.where(propensity_model.classes_ == label)[0][0]
                prop_vec[k] = prop_model_pred[k,label_idx]
                I[k] = 1 if 0 == A[k] and 0 == np.sum(A[one_hop[k]]) else 0
        w = I / prop_vec
        w_norm = w/np.sum(w)*N if np.sum(w) > 0 else 0
        psi = beta_hat + w_norm * (Y - beta_hat)
        psi_zero = psi.copy()
    else:
        psi_zero = np.zeros((N,))

    # Compute effects
    avg_psi_gamma = np.mean(psi_gamma)
    direct_effect = np.mean(psi_1_gamma) - np.mean(psi_0_gamma)
    spillover_effect = np.mean(psi_0_gamma) - np.mean(psi_zero)

    if return_raw:
        return {
            'psi_gamma': np.mean(psi_gamma, axis=1),
            'psi_zero': psi_zero,
            'psi_1_gamma': np.mean(psi_1_gamma, axis=1),
            'psi_0_gamma': np.mean(psi_0_gamma, axis=1),
        }

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