import numpy as np

def get_graph(num_nodes, max_neighbours, seed=None):
    """
    Generates a random graph with a specified number of nodes and maximum number of neighbours.
    
    Parameters:
    - num_nodes: int, the number of nodes in the graph
    - max_neighbours: int, the maximum number of neighbours each node can have
    - seed: int, optional seed for random number generation
    
    Returns:
    - graph: np.ndarray, adjacency matrix representing the graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create an empty adjacency matrix
    graph = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for i in range(num_nodes):
        # Randomly choose the number of neighbours for node i
        num_neighbours = np.random.randint(1, max_neighbours + 1)
        
        # Randomly select neighbours from the remaining nodes
        neighbours = np.random.choice(num_nodes, size=num_neighbours, replace=False)
        
        # Ensure no self-loops
        neighbours = neighbours[neighbours != i]
        
        # Update the adjacency matrix
        for neighbour in neighbours:
            graph[i][neighbour] = 1
            graph[neighbour][i] = 1  # Undirected graph
    
    return graph

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
    num_iter=1000,
    burnin=100,
    seed=42
):
    np.random.seed(seed)
    N = adj_matrix.shape[0]
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]

    # Initialize chains
    L_chain = np.zeros((num_iter, N, 3), dtype=int)
    A_chain = np.zeros((num_iter, N), dtype=int)
    Y_chain = np.zeros((num_iter, N), dtype=int)

    # Initialize variables
    L = np.random.binomial(1, 0.5, size=(N, 3))
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
    return {
        "Y_chain": Y_chain[burnin:],
        "A_chain": A_chain[burnin:],
        "L_chain": L_chain[burnin:]
    }