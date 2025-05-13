import numpy as np

def expit(x):
    return 1 / (1 + np.exp(-x))

def agc_effect(
    adj_matrix,
    tau, rho, nu, beta,
    treatment_allocation=0.5,
    R=10,
    burnin_R=5,
    seed=0
):
    np.random.seed(seed)
    N = adj_matrix.shape[0]
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]

    # Helper to sample Y and L
    def gibbs_sample_YL(tau, rho, nu, beta, R, burnin, Atype='all'):
        L_chain = np.zeros((R, N, 3))
        Y_chain = np.zeros((R, N), dtype=int)
        L = np.random.binomial(1, 0.5, size=(N, 3))
        Y = np.random.binomial(1, 0.5, size=N)
        
        for m in range(R):

            L_old = L.copy()
            for i in range(N):
                # Sample L[i, :]
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

            A = np.random.binomial(1, treatment_allocation, size=N)
            
            Y_old = Y.copy()
            for i in range(N):
                # Sample Y[i]
                if Atype == 'all':
                    linpred_Y = (
                        beta[0]
                        + beta[1] * A[i] + beta[2] * np.sum(A[neighbors[i]])
                        + beta[3] * L[i, 0] + beta[4] * np.sum(L_old[neighbors[i], 0])
                        + beta[5] * L[i, 1] + beta[6] * np.sum(L_old[neighbors[i], 1])
                        + beta[7] * L[i, 2] + beta[8] * np.sum(L_old[neighbors[i], 2])
                        + beta[9] * np.sum(Y_old[neighbors[i]])  # initialize with zeros
                    )
                elif Atype == 'ind_treat_1':
                    linpred_Y = (
                        beta[0]
                        + beta[1] * 1 + beta[2] * np.sum(A[neighbors[i]])
                        + beta[3] * L[i, 0] + beta[4] * np.sum(L_old[neighbors[i], 0])
                        + beta[5] * L[i, 1] + beta[6] * np.sum(L_old[neighbors[i], 1])
                        + beta[7] * L[i, 2] + beta[8] * np.sum(L_old[neighbors[i], 2])
                        + beta[9] * np.sum(Y_old[neighbors[i]])  # initialize with zeros
                    )
                elif Atype == 'ind_treat_0':
                    linpred_Y = (
                        beta[0]
                        + beta[1] * 0 + beta[2] * np.sum(A[neighbors[i]])
                        + beta[3] * L[i, 0] + beta[4] * np.sum(L_old[neighbors[i], 0])
                        + beta[5] * L[i, 1] + beta[6] * np.sum(L_old[neighbors[i], 1])
                        + beta[7] * L[i, 2] + beta[8] * np.sum(L_old[neighbors[i], 2])
                        + beta[9] * np.sum(Y_old[neighbors[i]])  # initialize with zeros
                    )
                elif Atype == 'all_0':
                    linpred_Y = (
                        beta[0]
                        + beta[1] * 0 + beta[2] * 0
                        + beta[3] * L[i, 0] + beta[4] * np.sum(L_old[neighbors[i], 0])
                        + beta[5] * L[i, 1] + beta[6] * np.sum(L_old[neighbors[i], 1])
                        + beta[7] * L[i, 2] + beta[8] * np.sum(L_old[neighbors[i], 2])
                        + beta[9] * np.sum(Y_old[neighbors[i]])  # initialize with zeros
                    )
                else:
                    raise ValueError("Invalid Atype. Choose from 'all', 'ind_treat_1', 'ind_treat_0', or 'all_0'.")
                Y[i] = np.random.binomial(1, expit(linpred_Y))

            L_chain[m] = L.copy()
            Y_chain[m] = Y.copy()
        return L_chain[burnin:], Y_chain[burnin:]

    psi_gamma, psi_zero, psi_1_gamma, psi_0_gamma = [], [], [], []
        
    # 1. Average outcome with treatment assigned at rate p
    L_chain, Y_chain = gibbs_sample_YL(tau, rho, nu, beta, R + burnin_R, burnin_R, Atype='all')
    for L, Y in zip(L_chain, Y_chain):
        psi_gamma.append(np.mean(Y))

    # 2. All control
    L_chain, Y_chain = gibbs_sample_YL(tau, rho, nu, beta, R + burnin_R, burnin_R, Atype='all_0')
    for L, Y in zip(L_chain, Y_chain):
        psi_zero.append(np.mean(Y))
    
    # 3. Individual treated, neighbors treated at rate p
    L_chain, Y_chain = gibbs_sample_YL(tau, rho, nu, beta, R + burnin_R, burnin_R, Atype='ind_treat_1')
    for L, Y in zip(L_chain, Y_chain):
        psi_1_gamma.append(np.mean(Y))
        
    # 4. Individual not treated, neighbors treated at rate p
    L_chain, Y_chain = gibbs_sample_YL(tau, rho, nu, beta, R + burnin_R, burnin_R, Atype='ind_treat_0')
    for L, Y in zip(L_chain, Y_chain):
        psi_0_gamma.append(np.mean(Y))
        
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


def agc_effect2(
    adj_matrix,
    tau, rho, nu, beta,
    treatment_allocation=0.5,
    R=10,
    burnin_R=5,
    seed=0
):
    np.random.seed(seed)
    N = adj_matrix.shape[0]
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]

    # Helper to sample Y and L
    def gibbs_sample_YL(tau, rho, nu, beta, R, burnin, Atype='all'):
        L_chain = np.zeros((R, N, 3))
        Y_chain = np.zeros((R, N), dtype=int)
        L = np.random.binomial(1, 0.5, size=(N, 3))
        Y = np.random.binomial(1, 0.5, size=N)
        
        for m in range(R):

            L_old = L.copy()
            for i in range(N):
                # Sample L[i, :]
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

            A = np.random.binomial(1, treatment_allocation, size=N)
            
            Y_old = Y.copy()
            for i in range(N):
                # Sample Y[i]
                #two_hop_neighbors = [q for p in neighbors[0] for q in neighbors[p] if q != 0]
                if Atype == 'all':
                    linpred_Y = (
                        beta[0]
                        + beta[1] * A[i] + beta[2] * np.sum(A[neighbors[i]])
                        + beta[3] * L[i, 0] * A[i] + beta[4] * np.sum(L[neighbors[i], 0]) * np.sum(A[neighbors[i]])
                        + beta[5] * L[i, 1] * A[i] + beta[6] * np.sum(L[neighbors[i], 1]) * np.sum(A[neighbors[i]])
                        + beta[7] * L[i, 2] * A[i] + beta[8] * np.sum(L[neighbors[i], 2]) * np.sum(A[neighbors[i]])
                        + beta[9] * A[i] * np.sum(A[neighbors[i]]) # initialize with zeros
                    )
                elif Atype == 'ind_treat_1':
                    linpred_Y = (
                        beta[0]
                        + beta[1] * 1 + beta[2] * np.sum(A[neighbors[i]])
                        + beta[3] * L[i, 0] * 1 + beta[4] * np.sum(L[neighbors[i], 0]) * np.sum(A[neighbors[i]])
                        + beta[5] * L[i, 1] * 1 + beta[6] * np.sum(L[neighbors[i], 1]) * np.sum(A[neighbors[i]])
                        + beta[7] * L[i, 2] * 1 + beta[8] * np.sum(L[neighbors[i], 2]) * np.sum(A[neighbors[i]])
                        + beta[9] * 1 * np.sum(A[neighbors[i]])  # initialize with zeros
                    )
                elif Atype == 'ind_treat_0':
                    linpred_Y = (
                        beta[0]
                        + beta[1] * 0 + beta[2] * np.sum(A[neighbors[i]])
                        + beta[3] * L[i, 0] * 0 + beta[4] * np.sum(L[neighbors[i], 0]) * np.sum(A[neighbors[i]])
                        + beta[5] * L[i, 1] * A[i] + beta[6] * np.sum(L[neighbors[i], 1]) * np.sum(A[neighbors[i]])
                        + beta[7] * L[i, 2] * A[i] + beta[8] * np.sum(L[neighbors[i], 2]) * np.sum(A[neighbors[i]])
                        + beta[9] * 0 * np.sum(A[neighbors[i]])  # initialize with zeros
                    )
                elif Atype == 'all_0':
                    linpred_Y = (
                        beta[0]
                        + beta[1] * 0 + beta[2] * 0
                        + beta[3] * L[i, 0] * 0 + beta[4] * np.sum(L[neighbors[i], 0]) * 0
                        + beta[5] * L[i, 1] * 0 + beta[6] * np.sum(L[neighbors[i], 1]) * 0
                        + beta[7] * L[i, 2] * 0 + beta[8] * np.sum(L[neighbors[i], 2]) * 0
                        + beta[9] * 0 * 0  # initialize with zeros
                    )
                else:
                    raise ValueError("Invalid Atype. Choose from 'all', 'ind_treat_1', 'ind_treat_0', or 'all_0'.")
                Y[i] = np.random.binomial(1, expit(linpred_Y))

            L_chain[m] = L.copy()
            Y_chain[m] = Y.copy()
        return L_chain[burnin:], Y_chain[burnin:]

    psi_gamma, psi_zero, psi_1_gamma, psi_0_gamma = [], [], [], []
        
    # 1. Average outcome with treatment assigned at rate p
    L_chain, Y_chain = gibbs_sample_YL(tau, rho, nu, beta, R + burnin_R, burnin_R, Atype='all')
    for L, Y in zip(L_chain, Y_chain):
        psi_gamma.append(np.mean(Y))

    # 2. All control
    L_chain, Y_chain = gibbs_sample_YL(tau, rho, nu, beta, R + burnin_R, burnin_R, Atype='all_0')
    for L, Y in zip(L_chain, Y_chain):
        psi_zero.append(np.mean(Y))
    
    # 3. Individual treated, neighbors treated at rate p
    L_chain, Y_chain = gibbs_sample_YL(tau, rho, nu, beta, R + burnin_R, burnin_R, Atype='ind_treat_1')
    for L, Y in zip(L_chain, Y_chain):
        psi_1_gamma.append(np.mean(Y))
        
    # 4. Individual not treated, neighbors treated at rate p
    L_chain, Y_chain = gibbs_sample_YL(tau, rho, nu, beta, R + burnin_R, burnin_R, Atype='ind_treat_0')
    for L, Y in zip(L_chain, Y_chain):
        psi_0_gamma.append(np.mean(Y))
        
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