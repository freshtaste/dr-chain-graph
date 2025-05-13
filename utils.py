# Run multiprocessing
import multiprocessing as mp
import traceback


def run_pll(f, args, processes=40):
    print('Multiprocessing {0} in {1} tasks, with {2} processes...'.format(f, len(args), processes))

    res_list = []
    error_list = []

    def log_result(res):
        res_list.append(res)

    def log_error(error):
        try:
            raise error
        except Exception:
            print(traceback.format_exc())
        error_list.append(error)

    pool = mp.Pool(processes=processes)
    for arg in args:
        pool.apply_async(f, kwds=arg, callback=log_result, error_callback=log_error)
    pool.close()
    pool.join()

    if len(error_list):
       print(f'{len(error_list)} jobs failed. ')

    print('Multiprocessing finished.')
    return res_list


from run_pll import cols_raw
from drnet import compute_avg_effects_std_from_raw
import numpy as np
from tqdm import tqdm

def compute_stats(res_list_array_dr1, res_list_array_dr2, ground_truth, adj_matrix):
    """
    Compute statistics from the results list. 
    res_list_array_dr: (n_sim, N, n_rep, 4) -> (n_sim, N, 4)
    """
    # compute the mean and standard deviation of the estimates
    effects_to_consider = ['average', 'direct', 'indirect', 'spillover_effect']

    coverage = np.zeros((res_list_array_dr1.shape[0], len(effects_to_consider)))
    estimates = np.zeros((res_list_array_dr1.shape[0], len(effects_to_consider)))
    std_hac = np.zeros((res_list_array_dr1.shape[0], len(effects_to_consider)))
    for i in tqdm(range(res_list_array_dr1.shape[0])):
        res_avg = res_list_array_dr1[i,:,cols_raw.index('psi_gamma')]
        res_direct = res_list_array_dr1[i,:,cols_raw.index('psi_1_gamma')] - res_list_array_dr1[i,:,cols_raw.index('psi_0_gamma')]
        res_indirect = res_list_array_dr1[i,:,cols_raw.index('psi_0_gamma')] - res_list_array_dr2[i,:,cols_raw.index('psi_0_gamma')]
        res_spillover = res_list_array_dr1[i,:,cols_raw.index('psi_0_gamma')] - res_list_array_dr1[i,:,cols_raw.index('psi_zero')]
        

        avg_effects_avg, se_hac_avg = compute_avg_effects_std_from_raw(res_avg, adj_matrix)
        avg_effects_direct, se_hac_direct = compute_avg_effects_std_from_raw(res_direct, adj_matrix)
        avg_effects_indirect, se_hac_indirect = compute_avg_effects_std_from_raw(res_indirect, adj_matrix)
        avg_effects_spillover, se_hac_spillover = compute_avg_effects_std_from_raw(res_spillover, adj_matrix)

        # coverage
        coverage[i,0] = (avg_effects_avg - 1.96*se_hac_avg < ground_truth['average']) & (ground_truth['average'] < avg_effects_avg + 1.96*se_hac_avg)
        coverage[i,1] = (avg_effects_direct - 1.96*se_hac_direct < ground_truth['direct']) & (ground_truth['direct'] < avg_effects_direct + 1.96*se_hac_direct)
        coverage[i,2] = (avg_effects_indirect - 1.96*se_hac_indirect < ground_truth['indirect']) & (ground_truth['indirect'] < avg_effects_indirect + 1.96*se_hac_indirect)
        coverage[i,3] = (avg_effects_spillover - 1.96*se_hac_spillover < ground_truth['spillover_effect']) & (ground_truth['spillover_effect'] < avg_effects_spillover + 1.96*se_hac_spillover)

        # estimates
        estimates[i,0] = avg_effects_avg
        estimates[i,1] = avg_effects_direct
        estimates[i,2] = avg_effects_indirect
        estimates[i,3] = avg_effects_spillover
        
        # std
        std_hac[i,0] = se_hac_avg
        std_hac[i,1] = se_hac_direct
        std_hac[i,2] = se_hac_indirect
        std_hac[i,3] = se_hac_spillover
    
    # compute stats
    coverage_rate = np.mean(coverage, axis=0)
    true_effect = np.array([ground_truth['average'], ground_truth['direct'], ground_truth['indirect'], ground_truth['spillover_effect']])
    bias = np.mean(estimates, axis=0) - true_effect
    mse = np.mean((estimates - true_effect)**2, axis=0)
    var = np.var(estimates, axis=0)
    ci_length = 2*1.96*np.mean(std_hac, axis=0)

    return {
        'coverage_rate': coverage_rate,
        'bias': bias,
        'mse': mse,
        'var': var,
        'ci_length': ci_length,
        'true_effect': true_effect,
    }