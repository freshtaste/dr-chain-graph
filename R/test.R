library("autognet")


get_graph <- function(N, min_degree = 2, max_degree = 5, seed = 0) {
  set.seed(seed)
  adj_matrix <- matrix(0, N, N)
  degree <- rep(0, N)
  degrees <- sample(min_degree:max_degree, N, replace = TRUE)
  
  # Candidate edges (i < j to avoid duplicates)
  candidates <- t(combn(N, 2))
  candidates <- candidates[sample(nrow(candidates)), , drop = FALSE]
  
  for (k in 1:nrow(candidates)) {
    i <- candidates[k, 1]
    j <- candidates[k, 2]
    if (degree[i] < degrees[i] && degree[j] < degrees[j]) {
      adj_matrix[i, j] <- 1
      adj_matrix[j, i] <- 1
      degree[i] <- degree[i] + 1
      degree[j] <- degree[j] + 1
    }
  }
  
  # Fix nodes with degree < min_degree
  for (i in 1:N) {
    while (degree[i] < degrees[i]) {
      possible <- which(adj_matrix[i, ] == 0 & degree < max_degree & seq_len(N) != i)
      if (length(possible) == 0) break
      j <- sample(possible, 1)
      adj_matrix[i, j] <- 1
      adj_matrix[j, i] <- 1
      degree[i] <- degree[i] + 1
      degree[j] <- degree[j] + 1
    }
  }
  
  return(adj_matrix)
}


simulate_network_causal_effects <- function(tau, rho, nu, beta, adjmat,
                                            group_lengths, group_functions,
                                            R = 50, burnin_cov = 10, burnin_R = 10,
                                            treatment_allocation = 0.5,
                                            subset = NULL,
                                            average = TRUE) {
  stopifnot(is.matrix(adjmat), is.numeric(tau), is.numeric(rho), is.numeric(beta))
  
  ncov <- length(tau)
  nrho <- choose(ncov, 2)
  N <- nrow(adjmat)
  weights <- rowSums(adjmat)
  
  # Format adjacency list (0-indexed for C++)
  adjacency <- lapply(1:N, function(i) which(adjmat[i, ] == 1) - 1)
  
  # Format nu as matrix if needed
  if (is.vector(nu) && length(nu) == ncov) {
    nu_mat <- diag(nu)
    additional_nu <- 0
  } else {
    nu_mat <- matrix(nu, nrow = ncov, byrow = TRUE)
    additional_nu <- 1
  }
  
  # Format rho as symmetric matrix
  rho_mat <- matrix(0, nrow = ncov, ncol = ncov)
  rho_mat[lower.tri(rho_mat, diag = FALSE)] <- rho
  rho_mat <- rho_mat + t(rho_mat)
  
  # Generate initial covariate values randomly
  cov_mat_start <- matrix(rbinom(ncov * N, 1, 0.5), nrow = N, ncol = ncov)
  
  # Simulate covariates using Gibbs
  cov_list <- networkGibbsOutCovCpp(
    tau, rho, nu_mat, ncov, R + burnin_R, N, burnin_cov, rho_mat,
    adjacency, weights, cov_mat_start, group_lengths, group_functions, additional_nu
  )
  
  # Subset of units to include (default: all)
  if (is.null(subset)) {
    subset <- 1:N
  }
  
  # Simulate E[Y_i(A)] under Bernoulli(p) assignment
  psi_p <- networkGibbsOuts1Cpp(cov_list, beta, treatment_allocation,
                                ncov, R + burnin_R, N, adjacency, weights,
                                burnin_R, as.numeric(average))[subset]
  
  # Simulate E[Y_i(0)] under no one treated
  psi_0 <- networkGibbsOuts1Cpp(cov_list, beta, 0,
                                ncov, R + burnin_R, N, adjacency, weights,
                                burnin_R, as.numeric(average))[subset]
  
  # Direct effect: E[Y_i(1, A_-i)] - E[Y_i(0, A_-i)]
  psi_1_gamma <- networkGibbsOuts2Cpp(cov_list, beta, treatment_allocation,
                                      ncov, R + burnin_R, N, adjacency, weights,
                                      subset, treatment_value = 1,
                                      burnin_R, as.numeric(average))
  psi_0_gamma <- networkGibbsOuts2Cpp(cov_list, beta, treatment_allocation,
                                      ncov, R + burnin_R, N, adjacency, weights,
                                      subset, treatment_value = 0,
                                      burnin_R, as.numeric(average))
  
  # Return average, direct, and spillover
  return(data.frame(
    average = mean(psi_p),
    direct = mean(psi_1_gamma - psi_0_gamma),
    spillover = mean(psi_0_gamma - psi_0)
  ))
}



# Load your compiled autognet package here
# library(autognet)

# Step 1: Create network
adj <- get_graph(800, min_degree = 2, max_degree = 4, seed = 2)

# Step 2: Define parameters
tau <- c(-1.0, 0.5, -0.5)  # length 3
rho <- c(0.1, 0.2, 0.1)    # lower triangle of rho matrix (3x3): (2,1), (3,1), (3,2)
nu <- matrix(c(0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0), nrow = 3, byrow = TRUE)  # 3x3
beta <- c(-0.3, -0.6, -0.2, -0.2, -0.05, -0.1, -0.01, 0.4, 0.01, 0.2)  # 10 values

# Step 3: Covariate structure for 3 binary covariates
group_lengths <- rep(1, 3)
group_functions <- rep(1, 3)

# Step 4: Simulate effects
results <- simulate_network_causal_effects(
  tau = tau,
  rho = rho,
  nu = nu,
  beta = beta,
  adjmat = adj,
  group_lengths = group_lengths,
  group_functions = group_functions,
  R = 50,
  burnin_cov = 10,
  burnin_R = 10,
  treatment_allocation = 0.7
)

# Step 5: View result
print(results)

