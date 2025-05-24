import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import binom

def expit(x):
    return 1 / (1 + np.exp(-x))

def get_neighbor_summary(X, adj_matrix):
    return adj_matrix @ X

def get_neighbor_summary2(X, adj_matrix):
    """
    Compute the average value of neighbors' attributes for each individual.
    
    Parameters:
    - X: (N x K) matrix of covariates (N individuals, K variables)
    - adj_matrix: (N x N) adjacency matrix (binary)
    
    Returns:
    - (N x K) matrix of neighbors' average covariates
    """
    # Number of neighbors for each person (vector of length N)
    num_neighbors = adj_matrix.sum(axis=1)
    
    # Avoid division by zero (if someone has zero neighbors, keep the average as zero)
    num_neighbors_safe = np.where(num_neighbors == 0, 1, num_neighbors)
    
    # Compute average
    neighbor_avg = (adj_matrix @ X) / num_neighbors_safe[:, np.newaxis]
    
    # Optional: set averages to zero for individuals with zero neighbors
    neighbor_avg[num_neighbors == 0, :] = 0
    
    return neighbor_avg


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
        L[:, 3], L_nb[:, 3],
        L[:, 4], L_nb[:, 4],
        Y_nb
    ])

def build_design_matrix_A(L, A, adj_matrix):
    L_nb = get_neighbor_summary(L, adj_matrix)
    A_nb = get_neighbor_summary(A.reshape(-1, 1), adj_matrix).flatten()
    return np.column_stack([
        L[:, 0], L_nb[:, 0],
        L[:, 1], L_nb[:, 1],
        L[:, 2], L_nb[:, 2],
        L[:, 3], L_nb[:, 3],
        L[:, 4], L_nb[:, 4],
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

def get_numerator_pi_em_new(a_mat_dict, A, GL, neighbours, neighbours_2hop, gamma, adj_matrix):
    N = adj_matrix.shape[0]
    
    max_neighbours = adj_matrix.sum(axis=1).max()
    numerator = np.zeros((N, 2, max_neighbours+1))
    for i in range(N):
        ni = [i]+neighbours[i]
        if len(ni) > 1:
            a_mat = a_mat_dict[len(neighbours[i])]
            vec_n = np.zeros((len(ni), a_mat.shape[1]))
            vec_n[1:, :] = a_mat.copy()
        else:
            vec_n = np.zeros((1,1))
        
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
            
            GL_neighbour_i = GL[ni].dot(vec_n)
            num_vec_i = np.exp(GL_neighbour_i + gamma[-1]*aa_n_i + gamma[-1]*aa_out_i)
            
            num_neighbours = len(neighbours[i])
            vec_treated_neighbours = vec_n[1:].sum(axis=0)
            for g in range(num_neighbours+1):
                # average over reps with num_neigh==g
                numerator[i, a, g] = np.sum(num_vec_i[vec_treated_neighbours==g])
    return numerator

def get_norm_constant_new(a_mat_dict, A, GL, neighbours, neighbours_2hop, gamma, adj_matrix, n_rep=1000):
    # compute denominator
    N = adj_matrix.shape[0]
    denominator = np.zeros((N,))
    for i in range(N):
        ni = [i]+neighbours[i]
        vec_n = a_mat_dict[len(ni)]
        
        # compute outter product to get a thousand 10 by 10
        mat_n = np.einsum('ik,jk->ijk', vec_n, vec_n)
        adj_max_n = adj_matrix[ni, :][:, ni]
        aa_n_i = (mat_n * adj_max_n[:, :, None]).sum(axis=(0, 1))/2
        
        nout = neighbours_2hop[i]
        vec_n_out = A[nout] # 790 by 1
        mat_n_out = np.einsum('ik,j->ijk', vec_n, vec_n_out)
        adj_max_n_out = adj_matrix[ni, :][:, nout]
        aa_out_i = (mat_n_out * adj_max_n_out[:, :, None]).sum(axis=(0, 1))

        GL_neighbour_i = GL[ni].dot(vec_n)
        num_vec_i = np.exp(GL_neighbour_i + gamma[-1]*aa_n_i + gamma[-1]*aa_out_i)
    
        denominator[i] = num_vec_i.sum()
        
    return denominator

from itertools import product

def generate_all_binary_vectors(max_n):
    binary_dict = {}
    for n in range(1, max_n + 1):
        # Generate all binary combinations
        combinations = list(product([0, 1], repeat=n))
        # Transpose to get shape (n, 2^n)
        a_mat = np.array(combinations).T
        binary_dict[n] = a_mat
    return binary_dict

def doubly_robust_em(A, L, Y, adj_matrix, treatment_allocation=0.7, seed=1, return_raw=False):
    np.random.seed(seed)
    
    # fit models
    L_a, L_y = L.copy(), L.copy()
    X_y = build_design_matrix_Y(A, L_y, Y, adj_matrix)
    model_y = fit_logistic_model(X_y, Y)
    X_a = build_design_matrix_A(L_a, A, adj_matrix)
    model_a = fit_logistic_model(X_a, A)
    
    # compute pi
    gamma = np.concatenate([model_a.intercept_, model_a.coef_.flatten()])
    N = adj_matrix.shape[0]
    neighbours, neighbours_2hop = get_2hop_neighbors(adj_matrix)
    L_nb = get_neighbor_summary(L_a, adj_matrix)
    GL = gamma[0] + L.dot(np.array([gamma[1], gamma[3], gamma[5], gamma[7], gamma[9]])) \
        + L_nb.dot(np.array([gamma[2], gamma[4], gamma[6], gamma[8], gamma[10]]))
    
    a_mat_dict = generate_all_binary_vectors(adj_matrix.sum(axis=1).max()+1)
    denominator = get_norm_constant_new(a_mat_dict, A, GL, neighbours, neighbours_2hop, gamma, adj_matrix)
    
    # compute the influence function
    numerator_vec = get_numerator_pi_em_new(a_mat_dict, A, GL, neighbours, neighbours_2hop, gamma, adj_matrix)
    pi_vec = numerator_vec / denominator[:, None, None] # N by 2 by max_neighbours

    beta_hat_vec = np.zeros(pi_vec.shape)
    for a in [0, 1]:
        for g in range(pi_vec.shape[2]):
            X_y_eval = X_y.copy()
            X_y_eval[:, 0] = a
            X_y_eval[:, 1] = g
            beta_hat_vec[:, a, g] = model_y.predict_proba(X_y_eval)[:, 1]

    A_nb = get_neighbor_summary(A.reshape(-1, 1), adj_matrix).flatten()
    psi_vec = np.zeros(pi_vec.shape)
    for a in [0, 1]:
        for g in range(pi_vec.shape[2]):
            I = ((A == a) & (A_nb == g)).astype(int)
            pi = pi_vec[:, a, g].copy()
            pi[I==0] = 1
            beta_hat = beta_hat_vec[:, a, g].copy() 
            w = I / pi
            psi_vec[:,a,g] = beta_hat + w * (Y - beta_hat)

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