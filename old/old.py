import numpy as np
from tqdm import tqdm

def expit(x):
    return 1 / (1 + np.exp(-x))

def agc_effect(
    adj_matrix,
    tau, rho, nu, beta,
    treatment_allocation=0.5,
    R=10,
    burnin_R=5,
    num_As=100,
    seed=0
):
    np.random.seed(seed)
    N = adj_matrix.shape[0]
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]

    # Helper to sample Y and L
    def gibbs_sample_YL(A, tau, rho, nu, R, burnin, Atype='all'):
        L_chain = np.zeros((R, N, 3))
        Y_chain = np.zeros((R, N), dtype=int)
        L = np.random.binomial(1, 0.5, size=(N, 3))
        Y = np.random.binomial(1, 0.5, size=N)
        
        for m in range(R):
            for i in range(N):
                # Sample Y[i]
                if Atype == 'all':
                    linpred_Y = (
                        beta[0]
                        + beta[1] * A[i] + beta[2] * np.sum(A[neighbors[i]])
                        + beta[3] * L[i, 0] + beta[4] * np.sum(L[neighbors[i], 0])
                        + beta[5] * L[i, 1] + beta[6] * np.sum(L[neighbors[i], 1])
                        + beta[7] * L[i, 2] + beta[8] * np.sum(L[neighbors[i], 2])
                        + beta[9] * np.sum(Y[neighbors[i]])  # initialize with zeros
                    )
                elif Atype == 'ind_treat_1':
                    linpred_Y = (
                        beta[0]
                        + beta[1] * 1 + beta[2] * np.sum(A[neighbors[i]])
                        + beta[3] * L[i, 0] + beta[4] * np.sum(L[neighbors[i], 0])
                        + beta[5] * L[i, 1] + beta[6] * np.sum(L[neighbors[i], 1])
                        + beta[7] * L[i, 2] + beta[8] * np.sum(L[neighbors[i], 2])
                        + beta[9] * np.sum(Y[neighbors[i]])  # initialize with zeros
                    )
                elif Atype == 'ind_treat_0':
                    linpred_Y = (
                        beta[0]
                        + beta[1] * 0 + beta[2] * np.sum(A[neighbors[i]])
                        + beta[3] * L[i, 0] + beta[4] * np.sum(L[neighbors[i], 0])
                        + beta[5] * L[i, 1] + beta[6] * np.sum(L[neighbors[i], 1])
                        + beta[7] * L[i, 2] + beta[8] * np.sum(L[neighbors[i], 2])
                        + beta[9] * np.sum(Y[neighbors[i]])  # initialize with zeros
                    )
                elif Atype == 'all_0':
                    linpred_Y = (
                        beta[0]
                        + beta[1] * 0 + beta[2] * 0
                        + beta[3] * L[i, 0] + beta[4] * np.sum(L[neighbors[i], 0])
                        + beta[5] * L[i, 1] + beta[6] * np.sum(L[neighbors[i], 1])
                        + beta[7] * L[i, 2] + beta[8] * np.sum(L[neighbors[i], 2])
                        + beta[9] * np.sum(Y[neighbors[i]])  # initialize with zeros
                    )
                Y[i] = np.random.binomial(1, expit(linpred_Y))

                # Sample L[i, :]
                for k in range(3):
                    linpred = tau[k]
                    for l in range(3):
                        if l != k:
                            linpred += rho[k, l] * L[i, l]
                        linpred += nu[k, l] * np.sum(L[neighbors[i], l])
                    L[i, k] = np.random.binomial(1, expit(linpred))
            L_chain[m] = L.copy()
            Y_chain[m] = Y.copy()
        return L_chain[burnin:], Y_chain[burnin:]

    psi_gamma, psi_zero, psi_1_gamma, psi_0_gamma = [], [], [], []
    for i in tqdm(range(num_As)):
        # Sample A
        A = np.random.binomial(1, treatment_allocation, size=N)
        
        # 1. Average outcome with treatment assigned at rate p
        L_chain, Y_chain = gibbs_sample_YL(A, tau, rho, nu, R + burnin_R, burnin_R, Atype='all')
        for L, Y in zip(L_chain, Y_chain):
            psi_gamma.append(np.mean(Y_chain))

        # 2. All control
        L_chain, Y_chain = gibbs_sample_YL(A, tau, rho, nu, R + burnin_R, burnin_R, Atype='all_0')
        for L, Y in zip(L_chain, Y_chain):
            psi_zero.append(np.mean(Y_chain))
        
        # 3. Individual treated, neighbors treated at rate p
        L_chain, Y_chain = gibbs_sample_YL(A, tau, rho, nu, R + burnin_R, burnin_R, Atype='ind_treat_1')
        for L, Y in zip(L_chain, Y_chain):
            psi_1_gamma.append(np.mean(Y_chain))
            
        # 4. Individual not treated, neighbors treated at rate p
        L_chain, Y_chain = gibbs_sample_YL(A, tau, rho, nu, R + burnin_R, burnin_R, Atype='ind_treat_0')
        for L, Y in zip(L_chain, Y_chain):
            psi_0_gamma.append(np.mean(Y_chain))
        
    # Compute effects
    avg_psi_gamma = np.mean(psi_gamma)
    direct_effect = np.mean(psi_1_gamma) - np.mean(psi_0_gamma)
    spillover_effect = np.mean(psi_0_gamma) - np.mean(psi_zero)
    return {
        "average": avg_psi_gamma,
        "direct_effect": direct_effect,
        "spillover_effect": spillover_effect
    }