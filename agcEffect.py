import numpy as np

def expit(x):
    return 1 / (1 + np.exp(-x))

def agc_effect(
    adj_matrix,
    tau, rho, nu, gamma, beta,
    treatment_allocation=0.5,
    R=10,
    burnin_R=5,
    average=True,
    seed=0
):
    np.random.seed(seed)
    N = adj_matrix.shape[0]
    neighbors = [np.where(adj_matrix[i])[0] for i in range(N)]

    # Helper to sample L using covariate model
    def gibbs_sample_L(tau, rho, nu, R, burnin):
        L_chain = np.zeros((R, N, 3))
        L = np.random.binomial(1, 0.5, size=(N, 3))

        for m in range(R):
            for i in range(N):
                for k in range(3):
                    linpred = tau[k]
                    for l in range(3):
                        if l != k:
                            linpred += rho[k, l] * L[i, l]
                        linpred += nu[k, l] * np.sum(L[neighbors[i], l])
                    L[i, k] = np.random.binomial(1, expit(linpred))
            L_chain[m] = L.copy()

        return L_chain[burnin:]

    # Helper to simulate outcomes given fixed treatment
    def simulate_Y(L, A, beta):
        Y = np.zeros(N)
        for i in range(N):
            linpred_Y = (
                beta[0]
                + beta[1] * A[i] + beta[2] * np.sum(A[neighbors[i]])
                + beta[3] * L[i, 0] + beta[4] * np.sum(L[neighbors[i], 0])
                + beta[5] * L[i, 1] + beta[6] * np.sum(L[neighbors[i], 1])
                + beta[7] * L[i, 2] + beta[8] * np.sum(L[neighbors[i], 2])
                + beta[9] * np.sum(Y[neighbors[i]])  # initialize with zeros
            )
            Y[i] = np.random.binomial(1, expit(linpred_Y))
        return Y

    # Run Gibbs on covariates
    L_chain = gibbs_sample_L(tau, rho, nu, R + burnin_R, burnin_R)

    # Store results
    psi_gamma, psi_zero, psi_1_gamma, psi_0_gamma = [], [], [], []

    for L in L_chain:
        # 1. Average outcome with treatment assigned at rate p
        A_p = np.random.binomial(1, treatment_allocation, size=N)
        Y_p = simulate_Y(L, A_p, beta)
        psi_gamma.append(np.mean(Y_p))

        # 2. All control
        A_zero = np.zeros(N, dtype=int)
        Y_zero = simulate_Y(L, A_zero, beta)
        psi_zero.append(np.mean(Y_zero))

        # 3. Individual treated, neighbors treated at rate p
        A_ind_treat = np.random.binomial(1, treatment_allocation, size=N)
        psi_1_vals = []
        for i in range(N):
            A_i = A_ind_treat.copy()
            A_i[i] = 1
            Y_i = simulate_Y(L, A_i, beta)
            psi_1_vals.append(Y_i[i])
        psi_1_gamma.append(np.mean(psi_1_vals))

        # 4. Individual not treated, neighbors treated at rate p
        psi_0_vals = []
        for i in range(N):
            A_i = A_ind_treat.copy()
            A_i[i] = 0
            Y_i = simulate_Y(L, A_i, beta)
            psi_0_vals.append(Y_i[i])
        psi_0_gamma.append(np.mean(psi_0_vals))

    # Compute effects
    avg_psi_gamma = np.mean(psi_gamma)
    direct_effect = np.mean(psi_1_gamma) - np.mean(psi_0_gamma)
    spillover_effect = np.mean(psi_0_gamma) - np.mean(psi_zero)

    return {
        "average": avg_psi_gamma,
        "direct_effect": direct_effect,
        "spillover_effect": spillover_effect
    }
