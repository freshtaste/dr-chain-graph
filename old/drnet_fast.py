import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import logsumexp

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

def build_design_matrix_A(L, A, adj_matrix):
    L_nb = get_neighbor_summary(L, adj_matrix)
    A_nb = get_neighbor_summary(A.reshape(-1, 1), adj_matrix).flatten()
    return np.column_stack([
        L[:, 0], L_nb[:, 0],
        L[:, 1], L_nb[:, 1],
        L[:, 2], L_nb[:, 2],
        A_nb
    ])

def build_design_matrix_Y_shared(L, Y, adj_matrix):
    # Precompute components not dependent on A
    L_nb = get_neighbor_summary(L, adj_matrix)
    Y_nb = get_neighbor_summary(Y.reshape(-1, 1), adj_matrix).flatten()
    return L, L_nb, Y_nb

def build_design_matrix_Y_batch(A_batch, L, L_nb, Y_nb, adj_matrix):
    A_nb_batch = adj_matrix @ A_batch
    return np.stack([
        A_batch,
        A_nb_batch,
        np.tile(L[:, 0][:, None], A_batch.shape[1]), np.tile(L_nb[:, 0][:, None], A_batch.shape[1]),
        np.tile(L[:, 1][:, None], A_batch.shape[1]), np.tile(L_nb[:, 1][:, None], A_batch.shape[1]),
        np.tile(L[:, 2][:, None], A_batch.shape[1]), np.tile(L_nb[:, 2][:, None], A_batch.shape[1]),
        np.tile(Y_nb[:, None], A_batch.shape[1])
    ], axis=2).transpose(0, 2, 1)  # shape: (N, 9, num_rep)

def compute_beta_probs_batch(X_y_batch, model_y, Atype='ind_treat_1'):
    X_y_batch = X_y_batch.copy()
    if Atype == 'ind_treat_1':
        X_y_batch[:, 0, :] = 1
    elif Atype == 'ind_treat_0':
        X_y_batch[:, 0, :] = 0
    elif Atype == 'all_0':
        X_y_batch[:, 0, :] = 0
        X_y_batch[:, 1, :] = 0
    # else: 'all' leaves as-is
    num_rep = X_y_batch.shape[2]
    return np.stack([
        model_y.predict_proba(X_y_batch[:, :, i])[:, 1] for i in range(num_rep)
    ], axis=1)  # shape: (N, num_rep)

def get_numerator_pi_vec_fast(a_mat, A, GL, neighbours, gamma, adj_matrix, Atype='ind_treat_1'):
    N, num_rep = a_mat.shape
    aGL = (a_mat.T * GL).T
    GL_neighbour = np.array([np.sum(aGL[[i]+neighbours[i]], axis=0) for i in range(N)])
    
    aa_n, aa_out, I = np.zeros((N, num_rep)), np.zeros((N, num_rep)), np.zeros((N, num_rep))
    universe = np.arange(N)

    for i in range(N):
        ni = [i] + neighbours[i]
        nout = np.setdiff1d(universe, ni, assume_unique=True)
        vec_n = a_mat[ni]  # (len(ni), num_rep)
        
        if Atype == 'ind_treat_1':
            vec_n[0, :] = 1
        elif Atype == 'ind_treat_0':
            vec_n[0, :] = 0

        mat_n = np.einsum('ik,jk->ijk', vec_n, vec_n)
        adj_max_n = adj_matrix[ni, :][:, ni]
        aa_n[i] = (mat_n * adj_max_n[:, :, None]).sum(axis=(0, 1)) / 2

        vec_n_out = A[nout]
        mat_n_out = np.einsum('ik,j->ijk', vec_n, vec_n_out)
        adj_max_n_out = adj_matrix[ni, :][:, nout]
        aa_out[i] = (mat_n_out * adj_max_n_out[:, :, None]).sum(axis=(0, 1))

        I[i] = np.all(vec_n == A[ni].reshape(-1, 1), axis=0).astype(int)

    numerator = np.exp(GL_neighbour + gamma[7] * (aa_n + aa_out))
    return numerator, I

def get_norm_constant_fast(A, GL, neighbours, gamma, adj_matrix, n_rep=1000):
    N = adj_matrix.shape[0]
    a_mat = np.random.binomial(1, 0.5, size=(N, n_rep))
    aGL = (a_mat.T * GL).T
    GL_neighbour = np.array([np.sum(aGL[[i]+neighbours[i]], axis=0) for i in range(N)])
    aa_n, aa_out = np.zeros((N, n_rep)), np.zeros((N, n_rep))
    universe = np.arange(N)

    for i in range(N):
        ni = [i] + neighbours[i]
        nout = np.setdiff1d(universe, ni, assume_unique=True)
        vec_n = a_mat[ni]
        mat_n = np.einsum('ik,jk->ijk', vec_n, vec_n)
        adj_max_n = adj_matrix[ni, :][:, ni]
        aa_n[i] = (mat_n * adj_max_n[:, :, None]).sum(axis=(0, 1)) / 2

        vec_n_out = A[nout]
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

# This script defines optimized core utilities. You can plug this into your `doubly_robust` function.
# Let me know if you want me to also rewrite the full `doubly_robust` function using these improvements.
def doubly_robust_optimized(A, L, Y, adj_matrix, treatment_allocation=0.7, num_rep=1000, seed=1):
    np.random.seed(seed)
    N = len(Y)
    neighbours = [list(adj_matrix[i].nonzero()[0]) for i in range(N)]

    # Fit outcome and treatment models
    X_y = build_design_matrix_Y(A, L, Y, adj_matrix)
    model_y = fit_logistic_model(X_y, Y)
    X_a = build_design_matrix_A(L, A, adj_matrix)
    model_a = fit_logistic_model(X_a, A)
    gamma = np.concatenate([model_a.intercept_, model_a.coef_.flatten()])

    # Compute GL and reuse across tasks
    L_nb = get_neighbor_summary(L, adj_matrix)
    GL = gamma[0] + L.dot(np.array([gamma[1], gamma[3], gamma[5]])) + \
         L_nb.dot(np.array([gamma[2], gamma[4], gamma[6]]))

    # Compute normalization constant once
    denominator = get_norm_constant_fast(A, GL, neighbours, gamma, adj_matrix, n_rep=num_rep)

    # Sample treatment matrices
    a_mat = np.random.binomial(1, treatment_allocation, size=(N, num_rep))

    # Precompute shared covariate design components
    L, L_nb, Y_nb = build_design_matrix_Y_shared(L, Y, adj_matrix)

    def estimate_effect(Atype):
        numerator_vec, I = get_numerator_pi_vec_fast(a_mat, A, GL, neighbours, gamma, adj_matrix, Atype=Atype)
        pi_vec = numerator_vec / denominator[:, None]
        X_y_eval_batch = build_design_matrix_Y_batch(a_mat, L, L_nb, Y_nb, adj_matrix)
        beta_hat_batch = compute_beta_probs_batch(X_y_eval_batch, model_y, Atype=Atype)  # shape (N, num_rep)
        psi_batch = beta_hat_batch + I / pi_vec * (Y[:, None] - beta_hat_batch)
        return psi_batch.mean()

    # Estimate all required effects
    avg_psi_gamma = estimate_effect('all')
    psi_1_gamma = estimate_effect('ind_treat_1')
    psi_0_gamma = estimate_effect('ind_treat_0')

    # Special case for psi_zero with all-0 treatment vector
    a_zero = np.zeros((N, 1), dtype=int)
    numerator, I = get_numerator_pi_vec_fast(a_zero, A, GL, neighbours, gamma, adj_matrix, Atype='all_0')
    pi_zero_vec = numerator / denominator[:, None]
    X_y_eval = build_design_matrix_Y_batch(a_zero, L, L_nb, Y_nb, adj_matrix)[:, :, 0]
    beta_hat = model_y.predict_proba(X_y_eval)[:, 1]
    psi = beta_hat + I[:, 0] / pi_zero_vec[:, 0] * (Y - beta_hat)
    psi_zero = psi.mean()

    # Compute effects
    direct_effect = psi_1_gamma - psi_0_gamma
    spillover_effect = psi_0_gamma - psi_zero

    return {
        "average": avg_psi_gamma,
        "direct_effect": direct_effect,
        "spillover_effect": spillover_effect,
        "psi_1_gamma": psi_1_gamma,
        "psi_0_gamma": psi_0_gamma,
        "psi_zero": psi_zero,
    }
