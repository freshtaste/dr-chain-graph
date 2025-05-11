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

def get_numerator_pi(a_vector, A, GL, neighbours, gamma, adj_matrix, Atype='ind_treat_1'):
    N = adj_matrix.shape[0]
    aGL = a_vector * GL
        
    GL_neighbour = np.array([np.sum(aGL[[i]+neighbours[i]], axis=0) for i in range(N)])
    aa_n, aa_out = np.zeros(N), np.zeros(N)
    for i in range(N):
        ni = [i]+neighbours[i]
        vec_n = a_vector[ni].copy()
        if Atype == 'ind_treat_1':
            vec_n[0] = 1
        elif Atype == 'ind_treat_0':
            vec_n[0] = 0
        # compute outter product
        mat_n = np.outer(vec_n, vec_n)
        adj_max_n = adj_matrix[ni, :][:, ni]
        aa_n[i] = np.sum(mat_n[adj_max_n == 1])/2
        
        nout = list(set(range(N)) - set(ni))
        vec_n_out = A[nout].copy()
        mat_n_out = np.outer(vec_n, vec_n_out)
        adj_max_n_out = adj_matrix[ni, :][:, nout]
        aa_out[i] = np.sum(mat_n_out[adj_max_n_out == 1])
        
    numerator = np.exp(GL_neighbour + gamma[7]*aa_n + gamma[7]*aa_out)
    return numerator

def get_numerator_pi_vec(a_mat, A, GL, neighbours, gamma, adj_matrix, Atype='ind_treat_1'):
    N = adj_matrix.shape[0]
    aGL = (a_mat.T * GL).T
    
    GL_neighbour = np.array([np.sum(aGL[[i]+neighbours[i]], axis=0) for i in range(N)])
    aa_n, aa_out = np.zeros((N, a_mat.shape[1])), np.zeros((N, a_mat.shape[1]))
    I = np.zeros((N, a_mat.shape[1]))
    for i in range(N):
        ni = [i]+neighbours[i]
        vec_n = a_mat[ni].copy()
        if Atype == 'ind_treat_1':
            vec_n[0] = 1
        elif Atype == 'ind_treat_0':
            vec_n[0] = 0
        # compute outter product
        mat_n = np.einsum('ik,jk->ijk', vec_n, vec_n)
        adj_max_n = adj_matrix[ni, :][:, ni]
        aa_n[i] = (mat_n * adj_max_n[:, :, None]).sum(axis=(0, 1))/2
        
        nout = list(set(range(N)) - set(ni))
        vec_n_out = A[nout].copy()
        mat_n_out = np.einsum('ik,j->ijk', vec_n, vec_n_out)
        adj_max_n_out = adj_matrix[ni, :][:, nout]
        aa_out[i] = (mat_n_out * adj_max_n_out[:, :, None]).sum(axis=(0, 1))
        
        # compute indicator
        I[i] = np.all(vec_n == A[ni].reshape(-1,1), axis=0).astype(int)
    
    numerator = np.exp(GL_neighbour + gamma[7]*aa_n + gamma[7]*aa_out)
    return numerator, I

def get_norm_constant(A, GL, neighbours, gamma, adj_matrix, n_rep=1000):
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
        
        nout = list(set(range(N)) - set(ni))
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


def doubly_robust(A, L, Y, adj_matrix, treatment_allocation=0.7, num_rep=1000, seed=1, return_raw=False):
    np.random.seed(seed)
    
    # fit models
    X_y = build_design_matrix_Y(A, L, Y, adj_matrix)
    model_y = fit_logistic_model(X_y, Y)
    X_a = build_design_matrix_A(L, A, adj_matrix)
    model_a = fit_logistic_model(X_a, A)
    
    # compute pi
    gamma = np.concatenate([model_a.intercept_, model_a.coef_.flatten()])
    N = adj_matrix.shape[0]
    neighbours = [list(adj_matrix[i].nonzero()[0]) for i in range(N)]
    L_nb = get_neighbor_summary(L, adj_matrix)
    GL = gamma[0] + L.dot(np.array([gamma[1], gamma[3], gamma[5]])) \
        + L_nb.dot(np.array([gamma[2], gamma[4], gamma[6]]))
        
    denominator = get_norm_constant(A, GL, neighbours, gamma, adj_matrix)
    
    # compute the influence function
    a_mat = np.random.binomial(1, treatment_allocation, size=(Y.shape[0], num_rep))
    numerator_vec, I = get_numerator_pi_vec(a_mat, A, GL, neighbours, gamma, adj_matrix, Atype='all')
    pi_vec = numerator_vec / denominator[:, None]
    
    psi_gamma = np.zeros((N, num_rep))
    for i in range(num_rep):
        X_y_eval = build_design_matrix_Y(a_mat[:,i], L, Y, adj_matrix)
        beta_hat = compute_beta_probs(X_y_eval, model_y, Atype='all')
        psi = beta_hat + I[:,i] / pi_vec[:, i] * (Y - beta_hat)
        psi_gamma[:, i] = psi.copy()
    
    numerator, I = get_numerator_pi_vec(a_mat, A, GL, neighbours, gamma, adj_matrix, Atype='ind_treat_1')
    pi_1_vec = numerator / denominator[:, None]
    psi_1_gamma = np.zeros((N, num_rep))
    for i in range(num_rep):
        X_y_eval = build_design_matrix_Y(a_mat[:,i], L, Y, adj_matrix)
        beta_hat = compute_beta_probs(X_y_eval, model_y, Atype='ind_treat_1')
        psi = beta_hat + I[:,i] / pi_1_vec[:, i] * (Y - beta_hat)
        psi_1_gamma[:, i] = psi.copy()
        
    numerator, I = get_numerator_pi_vec(a_mat, A, GL, neighbours, gamma, adj_matrix, Atype='ind_treat_0')
    pi_0_vec = numerator / denominator[:, None]
    psi_0_gamma = np.zeros((N, num_rep))
    for i in range(num_rep):
        X_y_eval = build_design_matrix_Y(a_mat[:,i], L, Y, adj_matrix)
        beta_hat = compute_beta_probs(X_y_eval, model_y, Atype='ind_treat_0')
        psi = beta_hat + I[:,i] / pi_0_vec[:, i] * (Y - beta_hat) 
        psi_0_gamma[:, i] = psi.copy()
    
    a_mat = np.zeros((Y.shape[0],1))
    numerator, I = get_numerator_pi_vec(a_mat, A, GL, neighbours, gamma, adj_matrix, Atype='all_0')
    pi_zero_vec = numerator / denominator[:, None]
    psi_zero = np.zeros((N,))
    X_y_eval = build_design_matrix_Y(a_mat, L, Y, adj_matrix)
    beta_hat = compute_beta_probs(X_y_eval, model_y, Atype='all_0')
    psi =  beta_hat + I[:,0] / pi_zero_vec[:, 0] * (Y - beta_hat)
    psi_zero = psi.copy()
    
    # Compute effects
    avg_psi_gamma = np.mean(psi_gamma)
    direct_effect = np.mean(psi_1_gamma) - np.mean(psi_0_gamma)
    spillover_effect = np.mean(psi_0_gamma) - np.mean(psi_zero)

    if return_raw:
        return {
            'psi_gamma': psi_gamma,
            'psi_zero': np.repeat(psi_zero[:, None], num_rep, axis=1),
            'psi_1_gamma': psi_1_gamma,
            'psi_0_gamma': psi_0_gamma,
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


from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.regression.linear_model import OLS

def compute_avg_effects_std_from_raw(psi_vec):
    """
    Compute the average effects and their standard deviations from a given raw psi vector.
    """
    avg_effects = np.mean(psi_vec)
    psi_arr = np.mean(psi_vec, axis=1)
    # compute the HAC standard error by regression
    T = len(psi_arr)
    X = np.ones((T, 1))  # Constant regressor
    Y = psi_arr.copy()
    model = OLS(Y, X)
    results = model.fit()

    # Compute HAC variance with specified lag
    lags = int(4 * (T / 100) ** (2 / 9))  # Newey-West lag rule-of-thumb
    hac_cov = cov_hac(results, maxlags=lags)

    # Extract standard error
    se_hac = np.sqrt(hac_cov[0, 0])
    # theta_hat = np.mean(Y)
    # print(f"theta_hat: {theta_hat:.4f}")
    # print(f"HAC standard error: {se_hac:.4f}")

    return avg_effects, se_hac
    