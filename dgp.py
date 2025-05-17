import numpy as np
from tqdm import tqdm
import networkx as nx
from scipy.special import expit

def get_graph(N, m=2, max_degree=10, seed=0):
    """
    Modified Barabási-Albert model with maximum degree constraint.
    """
    np.random.seed(seed)
    G = nx.empty_graph(m)  # Start with m isolated nodes
    targets = list(range(m))
    repeated_nodes = list(targets)

    for new_node in range(m, N):
        # Filter eligible targets (nodes with degree < max_degree)
        eligible = [node for node in repeated_nodes if G.degree[node] < max_degree]
        eligible = list(set(eligible))  # Remove duplicates

        # If not enough eligible nodes, connect to random nodes
        if len(eligible) < m:
            targets = np.random.choice(list(G.nodes), size=m, replace=False)
        else:
            probs = np.array([G.degree[node] for node in eligible], dtype=float)
            probs = probs/probs.sum() if probs.sum() > 0 else np.ones(len(eligible))/len(eligible)
            targets = np.random.choice(eligible, size=m, replace=False, p=probs)

        for target in targets:
            G.add_edge(new_node, target)

        # Update repeated_nodes list
        repeated_nodes.extend(targets)
        repeated_nodes.extend([new_node] * m)

    return nx.to_numpy_array(G, dtype=int)

def sample_L(L, adj_matrix, tau, rho, nu):
    N = L.shape[0]
    L_old = L.copy()
    
    # Neighbor summaries for each latent feature
    L_nb_0 = adj_matrix @ L_old[:, 0]
    L_nb_1 = adj_matrix @ L_old[:, 1]
    L_nb_2 = adj_matrix @ L_old[:, 2]

    # Compute each linpred separately with explicit coefficients
    linpred_L1 = (
        tau[0]
        + rho[0, 1] * L_old[:, 1]
        + rho[0, 2] * L_old[:, 2]
        + nu[0, 0] * L_nb_0
        + nu[0, 1] * L_nb_1
        + nu[0, 2] * L_nb_2
    )
    linpred_L2 = (
        tau[1]
        + rho[1, 0] * L_old[:, 0]
        + rho[1, 2] * L_old[:, 2]
        + nu[1, 0] * L_nb_0
        + nu[1, 1] * L_nb_1
        + nu[1, 2] * L_nb_2
    )
    linpred_L3 = (
        tau[2]
        + rho[2, 0] * L_old[:, 0]
        + rho[2, 1] * L_old[:, 1]
        + nu[2, 0] * L_nb_0
        + nu[2, 1] * L_nb_1
        + nu[2, 2] * L_nb_2
    )

    # Convert to probabilities
    prob_L1 = expit(linpred_L1)
    prob_L2 = expit(linpred_L2)
    prob_L3 = expit(linpred_L3)

    # Sample each L[:, d] separately
    L[:, 0] = np.random.binomial(1, prob_L1)
    L[:, 1] = np.random.binomial(1, prob_L2)
    L[:, 2] = np.random.binomial(1, prob_L3)

    return L


def sample_A(A, L, adj_matrix, gamma):
    L_nb = adj_matrix @ L
    A_nb = adj_matrix @ A

    linpred = (
        gamma[0]
        + gamma[1] * L[:, 0] + gamma[2] * L_nb[:, 0]
        + gamma[3] * L[:, 1] + gamma[4] * L_nb[:, 1]
        + gamma[5] * L[:, 2] + gamma[6] * L_nb[:, 2]
        + gamma[7] * A_nb
    )
    probs = expit(linpred)
    A[:] = np.random.binomial(1, probs)
    return A

def sample_Y1(Y, A, L, adj_matrix, beta, Atype='all'):
    L_nb = adj_matrix @ L
    A_nb = adj_matrix @ A
    Y_nb = adj_matrix @ Y
    if Atype == 'all' or Atype == 'gen':
        A_self = A.copy()
    elif Atype == 'ind_treat_1':
        A_self = np.ones_like(A)
    elif Atype == 'ind_treat_0':
        A_self = np.zeros_like(A)
    elif Atype == 'all_0':
        A_self = np.zeros_like(A)
        A_nb = np.zeros_like(A)
    else:
        raise ValueError("Invalid Atype. Choose from 'all', 'ind_treat_1', 'ind_treat_0', or 'all_0'.")

    linpred = (
        beta[0]
        + beta[1] * A_self + beta[2] * A_nb
        + beta[3] * L[:, 0] + beta[4] * L_nb[:, 0]
        + beta[5] * L[:, 1] + beta[6] * L_nb[:, 1]
        + beta[7] * L[:, 2] + beta[8] * L_nb[:, 2]
        + beta[9] * Y_nb
    )
    probs = expit(linpred)
    Y[:] = np.random.binomial(1, probs)
    return Y

def sample_Y2(Y, A, L, adj_matrix, beta, Atype='all'):
    L_nb = adj_matrix @ L
    A_nb = adj_matrix @ A
    if Atype == 'all' or Atype == 'gen':
        A_self = A.copy()
    elif Atype == 'ind_treat_1':
        A_self = np.ones_like(A)
    elif Atype == 'ind_treat_0':
        A_self = np.zeros_like(A)
    elif Atype == 'all_0':
        A_self = np.zeros_like(A)
        A_nb = np.zeros_like(A)
    else:
        raise ValueError("Invalid Atype. Choose from 'all', 'ind_treat_1', 'ind_treat_0', or 'all_0'.")

    linpred = (
        beta[0]
        + beta[1] * A_self + beta[2] * A_nb
        + beta[3] * L[:, 0] * A + beta[4] * L_nb[:, 0] * A_nb
        + beta[5] * L[:, 1] * A + beta[6] * L_nb[:, 1] * A_nb
        + beta[7] * L[:, 2] * A + beta[8] * L_nb[:, 2] * A_nb
        + beta[9] * A * A_nb
    )
    probs = expit(linpred)
    Y[:] = np.random.binomial(1, probs)
    return Y

def sample_network_chain(
    adj_matrix,
    tau, rho, nu, gamma, beta,
    R=50, burnin_R=10, seed=42,
    sample_Y_func=None,
    Atype=('gen', 0.7)
):
    np.random.seed(seed)
    N = adj_matrix.shape[0]

    # Initialize chains
    L_chain = np.zeros((R + burnin_R, N, 3), dtype=int)
    A_chain = np.zeros((R + burnin_R, N), dtype=int)
    Y_chain = np.zeros((R + burnin_R, N), dtype=int)

    # Initial values
    L = np.random.binomial(1, 0.5, size=(N, 3))
    A = np.random.binomial(1, 0.5, size=N)
    Y = np.random.binomial(1, 0.5, size=N)

    # Default functions if not supplied
    if sample_Y_func is None:
        raise ValueError("You must provide a sampling function for Y (sample_Y_func).")

    for m in range(R + burnin_R):
        L = sample_L(L, adj_matrix, tau, rho, nu)
        if Atype[0] == 'gen':
            A = sample_A(A, L, adj_matrix, gamma)
        else:
            A = np.random.binomial(1, Atype[1], size=N)
        Y = sample_Y_func(Y, A, L, adj_matrix, beta, Atype=Atype[0])

        L_chain[m] = L.copy()
        A_chain[m] = A.copy()
        Y_chain[m] = Y.copy()

    return Y_chain[burnin_R:], A_chain[burnin_R:], L_chain[burnin_R:]


def agcEffect(
    adj_matrix,
    tau, rho, nu, beta,
    treatment_allocation=0.5,
    R=10,
    burnin_R=5,
    seed=0,
    sample_Y_func=sample_Y1,
):
    np.random.seed(seed)
    N = adj_matrix.shape[0]

    psi_gamma, psi_zero, psi_1_gamma, psi_0_gamma = [], [], [], []
    Y_chain, _, L_chain = sample_network_chain(adj_matrix, tau, rho, nu, None, beta, R=R, burnin_R=burnin_R, seed=seed, sample_Y_func=sample_Y_func, Atype=('all', treatment_allocation))
    for Y, L in zip(Y_chain, L_chain):
        psi_gamma.append(np.mean(Y))
    
    Y_chain, _, L_chain = sample_network_chain(adj_matrix, tau, rho, nu, None, beta, R=R, burnin_R=burnin_R, seed=seed, sample_Y_func=sample_Y_func, Atype=('ind_treat_1', treatment_allocation))
    for Y, L in zip(Y_chain, L_chain):
        psi_1_gamma.append(np.mean(Y))

    Y_chain, _, L_chain = sample_network_chain(adj_matrix, tau, rho, nu, None, beta, R=R, burnin_R=burnin_R, seed=seed, sample_Y_func=sample_Y_func, Atype=('ind_treat_0', treatment_allocation))
    for Y, L in zip(Y_chain, L_chain):
        psi_0_gamma.append(np.mean(Y))

    Y_chain, _, L_chain = sample_network_chain(adj_matrix, tau, rho, nu, None, beta, R=R, burnin_R=burnin_R, seed=seed, sample_Y_func=sample_Y_func, Atype=('all_0', treatment_allocation))
    for Y, L in zip(Y_chain, L_chain):
        psi_zero.append(np.mean(Y))
    
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