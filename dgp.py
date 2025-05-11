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

def sample_network(adj_matrix, 
                   tau,       # shape (3,)
                   rho,       # shape (3, 3), with 0s on the diagonal
                   nu,        # shape (3, 3)
                   gamma,     # shape (8,)
                   beta,      # shape (10,)
                   num_iter=1000, 
                   seed=42):
    np.random.seed(seed)
    N = adj_matrix.shape[0]
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]
    
    # Initialize variables
    L = np.random.binomial(1, 0.5, size=(N, 3))  # L[i] = (L1, L2, L3)
    A = np.random.binomial(1, 0.5, size=N)
    Y = np.random.binomial(1, 0.5, size=N)

    for m in range(num_iter):
        i = m % N

        # --- Sample L[i, :] ---
        for k in range(3):
            linpred = tau[k]
            for l in range(3):
                if l != k:
                    linpred += rho[k, l] * L[i, l]
                linpred += nu[k, l] * np.sum(L[neighbors[i], l])
            prob = expit(linpred)
            L[i, k] = np.random.binomial(1, prob)

        # --- Sample A[i] ---
        linpred_A = (
            gamma[0]
            + gamma[1] * L[i, 0] + gamma[2] * np.sum(L[neighbors[i], 0])
            + gamma[3] * L[i, 1] + gamma[4] * np.sum(L[neighbors[i], 1])
            + gamma[5] * L[i, 2] + gamma[6] * np.sum(L[neighbors[i], 2])
            + gamma[7] * np.sum(A[neighbors[i]])
        )
        A[i] = np.random.binomial(1, expit(linpred_A))

        # --- Sample Y[i] ---
        linpred_Y = (
            beta[0]
            + beta[1] * A[i] + beta[2] * np.sum(A[neighbors[i]])
            + beta[3] * L[i, 0] + beta[4] * np.sum(L[neighbors[i], 0])
            + beta[5] * L[i, 1] + beta[6] * np.sum(L[neighbors[i], 1])
            + beta[7] * L[i, 2] + beta[8] * np.sum(L[neighbors[i], 2])
            + beta[9] * np.sum(Y[neighbors[i]])
        )
        Y[i] = np.random.binomial(1, expit(linpred_Y))

    return Y, A, L

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
        for i in range(N):

            # --- Sample L[i, :] ---
            for k in range(3):
                linpred = tau[k]
                for l in range(3):
                    if l != k:
                        linpred += rho[k, l] * L[i, l]
                    linpred += nu[k, l] * np.sum(L[neighbors[i], l])
                L[i, k] = np.random.binomial(1, expit(linpred))

            # --- Sample A[i] ---
            linpred_A = (
                gamma[0]
                + gamma[1] * L[i, 0] + gamma[2] * np.sum(L[neighbors[i], 0])
                + gamma[3] * L[i, 1] + gamma[4] * np.sum(L[neighbors[i], 1])
                + gamma[5] * L[i, 2] + gamma[6] * np.sum(L[neighbors[i], 2])
                + gamma[7] * np.sum(A[neighbors[i]])
            )
            A[i] = np.random.binomial(1, expit(linpred_A))

            # --- Sample Y[i] ---
            linpred_Y = (
                beta[0]
                + beta[1] * A[i] + beta[2] * np.sum(A[neighbors[i]])
                + beta[3] * L[i, 0] + beta[4] * np.sum(L[neighbors[i], 0])
                + beta[5] * L[i, 1] + beta[6] * np.sum(L[neighbors[i], 1])
                + beta[7] * L[i, 2] + beta[8] * np.sum(L[neighbors[i], 2])
                + beta[9] * np.sum(Y[neighbors[i]])
            )
            Y[i] = np.random.binomial(1, expit(linpred_Y))

        # Save to chains
        L_chain[m] = L.copy()
        A_chain[m] = A.copy()
        Y_chain[m] = Y.copy()

    # Apply burn-in
    return  Y_chain[burnin_R:], A_chain[burnin_R:], L_chain[burnin_R:]