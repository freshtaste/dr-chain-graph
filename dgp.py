import numpy as np
from tqdm import tqdm

def get_graph(N, min_degree=2, max_degree=5, seed=0):
    """
    Generate a symmetric adjacency matrix of an undirected graph where each node has:
      - at least `min_degree` neighbors
      - at most `max_degree` neighbors
    """
    np.random.seed(seed)
    adj_matrix = np.zeros((N, N), dtype=int)
    degree = np.zeros(N, dtype=int)
    degrees = np.random.randint(min_degree, max_degree + 1, size=N)

    # Candidate edges (i < j to avoid duplicates in symmetric matrix)
    candidates = [(i, j) for i in range(N) for j in range(i + 1, N)]
    np.random.shuffle(candidates)

    for (i, j) in candidates:
        if degree[i] < degrees[i] and degree[j] < degrees[j]:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
            degree[i] += 1
            degree[j] += 1

    # Fix nodes with degree < min_degree by connecting to random nodes
    for i in range(N):
        while degree[i] < degrees[i]:
            # Potential new connections: not self, not already connected, target degree < max
            possible = [j for j in range(N)
                        if j != i and adj_matrix[i, j] == 0 and degree[j] < max_degree]
            if not possible:
                break  # cannot fix further due to global constraints
            j = np.random.choice(possible)
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
            degree[i] += 1
            degree[j] += 1

    return adj_matrix

def expit(x):
    return 1 / (1 + np.exp(-x))

def sample_network_chain(
    adj_matrix,
    tau,       # shape (3,)
    rho,       # shape (3, 3), with 0s on the diagonal
    nu,        # shape (3, 3)
    gamma,     # shape (8,)
    beta,      # shape (10,)
    R=10,
    burnin_R=100,
    seed=42
):
    np.random.seed(seed)
    N = adj_matrix.shape[0]
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]

    # Initialize chains
    L_chain = np.zeros((R+burnin_R, N, 3), dtype=int)
    A_chain = np.zeros((R+burnin_R, N), dtype=int)
    Y_chain = np.zeros((R+burnin_R, N), dtype=int)

    # Initialize variables
    L = np.random.binomial(1, 0.5, size=(N, 3))
    A = np.random.binomial(1, 0.5, size=N)
    Y = np.random.binomial(1, 0.5, size=N)

    R = R + burnin_R  # Total iterations including burn-in
    for m in tqdm(range(R)):

        L_old = L.copy()
        for i in range(N):
            # --- Sample L[i, :] ---
            linpred_L1 = tau[0] + rho[0, 1] * L[i, 1] + rho[0, 2] * L[i, 2]\
                    + nu[0, 0] * np.sum(L_old[neighbors[i], 0]) \
                    + nu[0, 1] * np.sum(L_old[neighbors[i], 1]) \
                    + nu[0, 2] * np.sum(L_old[neighbors[i], 2])
            linpred_L2 = tau[1] + rho[1, 0] * L[i, 0] + rho[1, 2] * L[i, 2]\
                + nu[1, 0] * np.sum(L_old[neighbors[i], 0]) \
                + nu[1, 1] * np.sum(L_old[neighbors[i], 1]) \
                + nu[1, 2] * np.sum(L_old[neighbors[i], 2])
            linpred_L3 = tau[2] + rho[2, 0] * L[i, 0] + rho[2, 1] * L[i, 1]\
                + nu[2, 0] * np.sum(L_old[neighbors[i], 0]) \
                + nu[2, 1] * np.sum(L_old[neighbors[i], 1]) \
                + nu[2, 2] * np.sum(L_old[neighbors[i], 2])
                
            prob_L1 = expit(linpred_L1)
            prob_L2 = expit(linpred_L2)
            prob_L3 = expit(linpred_L3)
            L[i, 0] = np.random.binomial(1, prob_L1)
            L[i, 1] = np.random.binomial(1, prob_L2)
            L[i, 2] = np.random.binomial(1, prob_L3) 
        

        for i in range(N):
            # --- Sample A[i] ---
            linpred_A = (
                gamma[0]
                + gamma[1] * L[i, 0] + gamma[2] * np.sum(L[neighbors[i], 0])
                + gamma[3] * L[i, 1] + gamma[4] * np.sum(L[neighbors[i], 1])
                + gamma[5] * L[i, 2] + gamma[6] * np.sum(L[neighbors[i], 2])
                + gamma[7] * np.sum(A[neighbors[i]])
            )
            A[i] = np.random.binomial(1, expit(linpred_A))

        Y_old = Y.copy()
        for i in range(N):
            # --- Sample Y[i] ---
            linpred_Y = (
                beta[0]
                + beta[1] * A[i] + beta[2] * np.sum(A[neighbors[i]])
                + beta[3] * L[i, 0] + beta[4] * np.sum(L[neighbors[i], 0])
                + beta[5] * L[i, 1] + beta[6] * np.sum(L[neighbors[i], 1])
                + beta[7] * L[i, 2] + beta[8] * np.sum(L[neighbors[i], 2])
                + beta[9] * np.sum(Y_old[neighbors[i]])
            )
            Y[i] = np.random.binomial(1, expit(linpred_Y))

        # Save to chains
        L_chain[m] = L.copy()
        A_chain[m] = A.copy()
        Y_chain[m] = Y.copy()

    # Apply burn-in
    return  Y_chain[burnin_R:], A_chain[burnin_R:], L_chain[burnin_R:]


def sample_network_chain2(
    adj_matrix,
    tau,       # shape (3,)
    rho,       # shape (3, 3), with 0s on the diagonal
    nu,        # shape (3, 3)
    gamma,     # shape (8,)
    beta,      # shape (10,)
    R=10,
    burnin_R=100,
    seed=42
):
    np.random.seed(seed)
    N = adj_matrix.shape[0]
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]

    # Initialize chains
    L_chain = np.zeros((R+burnin_R, N, 3), dtype=int)
    A_chain = np.zeros((R+burnin_R, N), dtype=int)
    Y_chain = np.zeros((R+burnin_R, N), dtype=int)

    # Initialize variables
    L = np.random.binomial(1, 0.5, size=(N, 3))
    A = np.random.binomial(1, 0.5, size=N)
    Y = np.random.binomial(1, 0.5, size=N)

    R = R + burnin_R  # Total iterations including burn-in
    for m in tqdm(range(R)):

        L_old = L.copy()
        for i in range(N):
            # --- Sample L[i, :] ---
            linpred_L1 = tau[0] + rho[0, 1] * L[i, 1] + rho[0, 2] * L[i, 2]\
                    + nu[0, 0] * np.sum(L_old[neighbors[i], 0]) \
                    + nu[0, 1] * np.sum(L_old[neighbors[i], 1]) \
                    + nu[0, 2] * np.sum(L_old[neighbors[i], 2])
            linpred_L2 = tau[1] + rho[1, 0] * L[i, 0] + rho[1, 2] * L[i, 2]\
                + nu[1, 0] * np.sum(L_old[neighbors[i], 0]) \
                + nu[1, 1] * np.sum(L_old[neighbors[i], 1]) \
                + nu[1, 2] * np.sum(L_old[neighbors[i], 2])
            linpred_L3 = tau[2] + rho[2, 0] * L[i, 0] + rho[2, 1] * L[i, 1]\
                + nu[2, 0] * np.sum(L_old[neighbors[i], 0]) \
                + nu[2, 1] * np.sum(L_old[neighbors[i], 1]) \
                + nu[2, 2] * np.sum(L_old[neighbors[i], 2])
                
            prob_L1 = expit(linpred_L1)
            prob_L2 = expit(linpred_L2)
            prob_L3 = expit(linpred_L3)
            L[i, 0] = np.random.binomial(1, prob_L1)
            L[i, 1] = np.random.binomial(1, prob_L2)
            L[i, 2] = np.random.binomial(1, prob_L3) 
        

        for i in range(N):
            # --- Sample A[i] ---
            linpred_A = (
                gamma[0]
                + gamma[1] * L[i, 0] + gamma[2] * np.sum(L[neighbors[i], 0])
                + gamma[3] * L[i, 1] + gamma[4] * np.sum(L[neighbors[i], 1])
                + gamma[5] * L[i, 2] + gamma[6] * np.sum(L[neighbors[i], 2])
                + gamma[7] * np.sum(A[neighbors[i]])
            )
            A[i] = np.random.binomial(1, expit(linpred_A))

        Y_old = Y.copy()
        for i in range(N):
            # --- Sample Y[i] ---
            #two_hop_neighbors = [q for p in neighbors[0] for q in neighbors[p] if q != 0]
            linpred_Y = (
                beta[0]
                + beta[1] * A[i] + beta[2] * np.sum(A[neighbors[i]])
                + beta[3] * L[i, 0] * A[i] + beta[4] * np.sum(L[neighbors[i], 0]) * np.sum(A[neighbors[i]])
                + beta[5] * L[i, 1] * A[i] + beta[6] * np.sum(L[neighbors[i], 1]) * np.sum(A[neighbors[i]])
                + beta[7] * L[i, 2] * A[i] + beta[8] * np.sum(L[neighbors[i], 2]) * np.sum(A[neighbors[i]])
                + beta[9] * A[i] * np.sum(A[neighbors[i]])
            )
            Y[i] = np.random.binomial(1, expit(linpred_Y))

        # Save to chains
        L_chain[m] = L.copy()
        A_chain[m] = A.copy()
        Y_chain[m] = Y.copy()

    # Apply burn-in
    return  Y_chain[burnin_R:], A_chain[burnin_R:], L_chain[burnin_R:]