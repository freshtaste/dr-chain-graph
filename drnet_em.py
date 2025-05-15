import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import binom

def expit(x):
    return 1 / (1 + np.exp(-x))

def get_neighbor_summary(X, adj_matrix):
    return adj_matrix @ X

def fit_logistic_model(X, y):
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
    model.fit(X, y)
    return model

def get_2hop_neighbors(adj_matrix):
    """
    Returns for each node:
      - one_hop: list of direct neighbors
      - two_hop: list of neighbors-of-neighbors excluding direct neighbors and the node itself
    """
    N = adj_matrix.shape[0]
    one_hop = [list(np.where(adj_matrix[i])[0]) for i in range(N)]
    two_hop = []
    for i in range(N):
        # gather neighbors of neighbors
        second = set()
        for j in one_hop[i]:
            second.update(list(np.where(adj_matrix[j])[0]))
        # remove direct neighbors and self
        second.difference_update(one_hop[i])
        second.discard(i)
        two_hop.append(sorted(second))
    return one_hop, two_hop

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

def get_numerator_pi_em(a_mat, A, GL, neighbours, neighbours_2hop, gamma, adj_matrix):
    N = adj_matrix.shape[0]
    num_rep = a_mat.shape[1]
    aGL = (a_mat.T * GL).T
    
    max_neighbours = adj_matrix.sum(axis=1).max()
    numerator = np.zeros((N, 2, max_neighbours+1))
    for i in range(N):
        ni = [i]+neighbours[i]
        vec_n = a_mat[ni].copy()
        
        # compute outter product
        for a in [0, 1]:
            vec_n[0] = a
            mat_n = np.einsum('ik,jk->ijk', vec_n, vec_n)
            adj_max_n = adj_matrix[ni, :][:, ni]
            aa_n_i = (mat_n * adj_max_n[:, :, None]).sum(axis=(0, 1))/2
            
            nout = neighbours_2hop[i]
            vec_n_out = A[nout] # 790 by 1
            mat_n_out = np.einsum('ik,j->ijk', vec_n, vec_n_out)
            adj_max_n_out = adj_matrix[ni, :][:, nout]
            aa_out_i = (mat_n_out * adj_max_n_out[:, :, None]).sum(axis=(0, 1))
            
            GL_neighbour_i = np.sum(aGL[neighbours[i]], axis=0) + GL[i]*a
            num_vec_i = np.exp(GL_neighbour_i + gamma[7]*aa_n_i + gamma[7]*aa_out_i)
            
            num_neighbours = len(neighbours[i])
            vec_treated_neighbours = vec_n[1:].sum(axis=0)
            for g in range(num_neighbours+1):
                # average over reps with num_neigh==g
                numerator[i, a, g] = np.sum(num_vec_i[vec_treated_neighbours==g])/num_rep*(2**num_neighbours)
    return numerator

def get_norm_constant(A, GL, neighbours, neighbours_2hop, gamma, adj_matrix, n_rep=1000):
    # compute denominator
    N = adj_matrix.shape[0]
    a_mat = np.random.binomial(1, 0.5, size=(N, n_rep))
    aGL = (a_mat.T * GL).T
    GL_neighbour = np.array([np.sum(aGL[[i]+neighbours[i]], axis=0) for i in range(N)])
    aa_n, aa_out = np.zeros((N, n_rep)), np.zeros((N, n_rep))
    for i in range(N):
        ni = [i]+neighbours[i]
        vec_n = a_mat[ni] # 10 by 1000
        # compute outter product to get a thousand 10 by 10
        mat_n = np.einsum('ik,jk->ijk', vec_n, vec_n)
        adj_max_n = adj_matrix[ni, :][:, ni]
        aa_n[i] = (mat_n * adj_max_n[:, :, None]).sum(axis=(0, 1))/2
        
        nout = neighbours_2hop[i]
        vec_n_out = A[nout] # 790 by 1
        mat_n_out = np.einsum('ik,j->ijk', vec_n, vec_n_out)
        adj_max_n_out = adj_matrix[ni, :][:, nout]
        aa_out[i] = (mat_n_out * adj_max_n_out[:, :, None]).sum(axis=(0, 1))
    
    denominator = np.exp(GL_neighbour + gamma[7]*aa_n + gamma[7]*aa_out)
    
    # approximate the sum in the denominator
    num_neighbours = np.array([len(neighbours[i]) for i in range(N)])
    group_size = num_neighbours + 1
    num_a_group = 2**group_size
    denominator = np.mean(denominator, axis=1) * num_a_group
    
    return denominator

def doubly_robust(A, L, Y, adj_matrix, treatment_allocation=0.7, num_rep=1000, seed=1, return_raw=False, psi_0_gamma_only=False,
                  mispec=None):
    np.random.seed(seed)
    
    # fit models
    L_a, L_y = L.copy(), L.copy()
    if mispec == 'outcome':
        L_y = np.random.binomial(1, 0.5, size=L_y.shape)
    elif mispec == 'treatment':
        L_a = np.random.binomial(1, 0.5, size=L_a.shape)
    X_y = build_design_matrix_Y(A, L_y, Y, adj_matrix)
    model_y = fit_logistic_model(X_y, Y)
    X_a = build_design_matrix_A(L_a, A, adj_matrix)
    model_a = fit_logistic_model(X_a, A)
    
    # compute pi
    gamma = np.concatenate([model_a.intercept_, model_a.coef_.flatten()])
    N = adj_matrix.shape[0]
    neighbours, neighbours_2hop = get_2hop_neighbors(adj_matrix)
    L_nb = get_neighbor_summary(L_a, adj_matrix)
    GL = gamma[0] + L.dot(np.array([gamma[1], gamma[3], gamma[5]])) \
        + L_nb.dot(np.array([gamma[2], gamma[4], gamma[6]]))
        
    denominator = get_norm_constant(A, GL, neighbours, neighbours_2hop, gamma, adj_matrix)
    
    # compute the influence function
    a_mat = np.random.binomial(1, treatment_allocation, size=(Y.shape[0], num_rep))
    numerator_vec = get_numerator_pi_em(a_mat, A, GL, neighbours, neighbours_2hop, gamma, adj_matrix)
    pi_vec = numerator_vec / denominator[:, None, None] # N by 2 by max_neighbours

    beta_hat_vec = np.zeros(pi_vec.shape)
    for a in [0, 1]:
        for g in range(pi_vec.shape[2]):
            X_y_eval = X_y.copy()
            X_y_eval[:, 0] = a
            X_y_eval[:, 1] = g
            beta_hat_vec[:, a, g] = model_y.predict_proba(X_y)[:, 1]

    A_nb = get_neighbor_summary(A.reshape(-1, 1), adj_matrix).flatten()
    psi_vec = np.zeros(pi_vec.shape)
    for a in [0, 1]:
        for g in range(pi_vec.shape[2]):
            I = ((A == a) & (A_nb == g)).astype(int)
            pi = pi_vec[:, a, g]
            beta_hat = beta_hat_vec[:, a, g]
            psi_vec[:,a,g] = beta_hat + I / pi * (Y - beta_hat)

    # compute all 1
    prob_allocation_vec = np.zeros((N, pi_vec.shape[2]))
    for i in range(N):
        num_neighbours = len(neighbours[i])
        prob_allocation_vec[i,:num_neighbours+1] = np.array([binom.pmf(k, num_neighbours, treatment_allocation) 
                                             for k in range(num_neighbours+1)])

    psi_gamma, psi_1_gamma, psi_0_gamma, psi_zero = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    psi_1_gamma = (psi_vec[:, 1, :]*prob_allocation_vec).sum(axis=1)
    psi_0_gamma = (psi_vec[:, 0, :]*prob_allocation_vec).sum(axis=1)
    psi_zero = psi_vec[:,0,0]
    psi_gamma = treatment_allocation * psi_1_gamma + (1-treatment_allocation) * psi_0_gamma

    if return_raw:
        return {
            'psi_gamma': psi_gamma,
            'psi_zero': psi_zero,
            'psi_1_gamma': psi_1_gamma,
            'psi_0_gamma': psi_0_gamma,
        }

    # Compute effects
    avg_psi_gamma = np.mean(psi_gamma)
    direct_effect = np.mean(psi_1_gamma) - np.mean(psi_0_gamma)
    spillover_effect = np.mean(psi_0_gamma) - np.mean(psi_zero)

    return {
        "average": avg_psi_gamma,
        "direct_effect": direct_effect,
        "spillover_effect": spillover_effect,
        "psi_1_gamma": np.mean(psi_1_gamma),
        "psi_0_gamma": np.mean(psi_0_gamma),
        "psi_zero": np.mean(psi_zero),
    }