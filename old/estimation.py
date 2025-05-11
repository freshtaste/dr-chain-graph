import numpy as np

def expit(x):
    return 1 / (1 + np.exp(-x))

def prepare_features_outcome_model(Y, A, L, adj_matrix):
    """
    Constructs the design matrix for logistic regression of Y_i on:
    A_i, sum(A_neighbors), L_i (3), sum(L_neighbors, 3), sum(Y_neighbors)
    """
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

    X = np.array(X_list)
    return X  # shape: (N, 10), matches beta[0] to beta[9] in model


import numpy as np
from sklearn.linear_model import LogisticRegression

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
    return hash((len(pattern), tuple(pattern)))

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
        pattern = np.concatenate(([A[i]], A[one_hop[i]]))
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
    X_y = prepare_features_outcome_model(Y, A, L, adj_matrix)
    outcome_model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=10000)
    outcome_model.fit(X_y, Y)

    # Propensity model
    X_p, y_p = prepare_features_propensity_model(A, L, adj_matrix)
    propensity_model = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=10000
    )
    propensity_model.fit(X_p, y_p)

    return {
        'outcome_model': outcome_model,
        'propensity_model': propensity_model
    }

def compute_influence_function(Y, A, L, adj_matrix, models, a):
    """
    Compute doubly robust influence values for each unit i.
    """
    N = adj_matrix.shape[0]
    one_hop, two_hop = get_2hop_neighbors(adj_matrix)
    outcome_model = models['outcome_model']
    propensity_model = models['propensity_model']

    psi = np.zeros(N)
    X_y = prepare_features_outcome_model(Y, a, L, adj_matrix)
    for i in range(N):
        # features for outcome
        X_y_i = X_y[i].reshape(1, -1)
        mu_hat = outcome_model.predict_proba(X_y_i)[0, 1]

        # features for propensity
        Li = L[i].reshape(1, -1)
        L1 = np.sum(L[one_hop[i]], axis=0).reshape(1, -1) if len(one_hop[i]) > 0 else np.zeros((1,3))
        L2 = np.sum(L[two_hop[i]], axis=0).reshape(1, -1) if len(two_hop[i]) > 0 else np.zeros((1,3))
        A2 = np.array([[np.sum(A[two_hop[i]])]]) if len(two_hop[i]) > 0 else np.zeros((1,1))
        X_p_i = np.hstack([Li, L1, L2, A2])
        # label index
        pattern = np.concatenate(([a[i]], a[one_hop[i]]))
        label = encode_pattern(pattern)
        label_idx = np.where(propensity_model.classes_ == label)[0][0]
        pi_hat = propensity_model.predict_proba(X_p_i)[0, label_idx]

        # indicator I{A_neighbors pattern}
        I = 1 if a[i] == A[i] and np.all(a[one_hop[i]] == A[one_hop[i]]) else 0

        psi[i] = I / pi_hat * (Y[i] - mu_hat) + mu_hat

    return psi
