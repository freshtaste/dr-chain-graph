from drnet import doubly_robust

def run_dr(A_chain, L_chain, Y_chain, adj, i):
    """
    Run doubly robust estimator
    """
    ret_i = doubly_robust(A_chain[i], L_chain[i], Y_chain[i], adj)
    
    return (ret_i['average'], ret_i['direct_effect'], ret_i['spillover_effect'], 
            ret_i['psi_0_gamma'], ret_i['psi_zero'], ret_i['psi_1_gamma'])