import numpy as np
from scipy.stats import beta
import copy
import numpy as np
from scipy.stats import beta
import mpmath as mp



def mcs_unknown(brs_u, probs, sys_fun_rs, cpms, sys_name, cov_t, sys_st_monitor, sys_st_prob, rand_seed=None):
    """
    Perform Monte Carlo simulation for the unknown state.
    """

    # Set the random seed
    if rand_seed:
        np.random.seed(rand_seed)

    brs_u_probs = [b[2] for b in brs_u]
    brs_u_prob = round(sum(brs_u_probs), 20)
    
    print(f'brs_u_probs: {brs_u_probs}')
    print(f"Number of brs_u_probs: {len(brs_u_probs)}")
    print(f'brs_u_prob: {brs_u_prob}')


    samples = []
    samples_sys = np.empty((0, 1), dtype=int)
    sample_probs = []

    nsamp, nfail = 0, 0
    pf, cov = 0.0, 1.0

    while cov > cov_t:

        nsamp += 1

        sample1 = {}
        s_prob1 = {}

        # select a branch
        p=[]
        
        for i in brs_u_probs:
            p.append(i / brs_u_prob)
        br_id = np.random.choice(range(len(brs_u)), p=p)
        br = brs_u[br_id]


        for e in br[0].keys():
            d = br[0][e]
            u = br[1][e]

            if d < u: # (fail, surv)
                st = np.random.choice(range(d, u + 1), p=[probs[e][d], probs[e][u]])
            else:
                st = d

            sample1[e] = st
            s_prob1[e] = probs[e][st]

        # system function run
        val, sys_st = sys_fun_rs(sample1)
        
        if sys_st == 's':
            sys_st = 1
        else:
            sys_st = 0
        
        samples.append(sample1)
        sample_probs.append(s_prob1)
        samples_sys = np.vstack((samples_sys, [sys_st]))

        if sys_st == sys_st_monitor:
            nfail += 1

        if nsamp > 9:
            prior = 0.01
            a, b = prior + nfail, prior + (nsamp-nfail) # Bayesian estimation assuming beta conjucate distribution

            pf_s = a / (a+b)
            var_s = a*b / (a+b)**2 / (a+b+1)
            std_s = np.sqrt(var_s)

            print(f"pf_s = {a:.10f} / ({a:.10f} + {b:.10f}) = {pf_s:.10f}\n")

            pf = sys_st_prob + brs_u_prob * pf_s
            std = brs_u_prob * std_s
            cov = std/pf

            print(f"pf = {sys_st_prob:.10f} + ({brs_u_prob:.10f} * {pf_s:.10f}) = {pf:.10f}\n")

            conf_p = 0.95 # confidence interval
            low = beta.ppf(0.5*(1-conf_p), a, b)
            up = beta.ppf(1 - 0.5*(1-conf_p), a, b)
            cint = sys_st_prob + brs_u_prob * np.array([low, up])

        if nsamp % 1000 == 0:
            print(f'nsamp: {nsamp}, pf: {pf:.4e}, cov: {cov:.4e}')


    # Allocate samples to CPMs
    Csys = np.zeros((nsamp, len(probs)), dtype=int)
    Csys = np.hstack((samples_sys, Csys))

    for i, v in enumerate(cpms[sys_name].variables[1:]):
        Cv = np.array([s[v.name] for s in samples], dtype=int).T
        cpms[v.name].Cs = Cv
        cpms[v.name].q = np.array([p[v.name] for p in sample_probs], dtype=float).T
        cpms[v.name].sample_idx = np.arange(nsamp, dtype=int)

        Csys[:, i + 1] = Cv.flatten()

    Csys = Csys.astype(int)

    cpms[sys_name].Cs = Csys
    cpms[sys_name].q = np.ones((nsamp, 1), dtype=float)
    cpms[sys_name].sample_idx = np.arange(nsamp, dtype=int)

    result = {'pf': pf, 'cov': cov, 'nsamp': nsamp, 'cint_low': cint[0], 'cint_up': cint[1]}

    return cpms, result



def eventspace_x0_filter(brs_u, X_n, P_Xn_0, sys_fun):
    """
    Step 1: Event-space Filtering + System Function Application + Survival/Unknown Branch Classification

    Parameters:
    - brs_u: List of unknown branches from the BRC output.
    - X_n: Target component (e.g., 'e5' for component 5).
    - P_Xn_0: Probability of X_n = 0.
    - sys_fun: System function that determines the system state ('s' for survival, 'f' for failure, 'u' for unknown).

    Returns:
    - survival_known_branch: List of branches confirmed as survival.
    - unknown_branch: List of branches classified as unknown.
    """
    lower0_upper0 = []  # Branches where both lower and upper states are 0
    lower0_upper1 = []  # Branches where lower state is 0 and upper state is 1
    lower0_upper1_filtered = []  # Modified branches where upper state is changed to 0

    # Classify branches based on their lower and upper states
    for branch in brs_u:
        lower_state = branch.down.get(X_n, None)  # Lower state of X_n
        upper_state = branch.up.get(X_n, None)    # Upper state of X_n

        if lower_state == 0 and upper_state == 0:
            lower0_upper0.append(branch)  # Directly classify as an unknown branch

        elif lower_state == 0 and upper_state == 1:
            lower0_upper1.append(branch)  # Store for further modification

            # Modify the branch: change upper state to 0 and adjust probability
            new_branch = copy.deepcopy(branch)
            new_branch.up[X_n] = 0  
            new_branch.p *= P_Xn_0  
            lower0_upper1_filtered.append(new_branch)

    # Apply system function to determine the system states
    lower0_upper1_filtered_aftersystemfcn = []
    survival_known_branch = []  
    unknown_branch = lower0_upper0.copy()  

    for branch in lower0_upper1_filtered:
        comps_st_lower = branch.down  
        comps_st_upper = branch.up    

        # Evaluate system states for both lower and upper bounds
        _, sys_st_lower, _ = sys_fun(comps_st_lower)  
        _, sys_st_upper, _ = sys_fun(comps_st_upper)  

        # Copy branch and update system states
        new_branch = copy.deepcopy(branch)
        new_branch.down_state = sys_st_lower  
        new_branch.up_state = sys_st_upper  

        # Convert to unknown branch if system state transitions from failure to survival
        if sys_st_upper == 's' and sys_st_lower == 'f':  
            new_branch.up_state = 'u'  
            new_branch.down_state = 'u'  

        lower0_upper1_filtered_aftersystemfcn.append(new_branch)

        # Classify as a confirmed survival branch if both system states indicate survival
        if sys_st_upper == 's' and sys_st_lower == 's':
            survival_known_branch.append(new_branch)

        # Classify as an unknown branch if both system states are unknown
        if (sys_st_upper == 'u' and sys_st_lower == 'u') or (new_branch.up_state == 'u' and new_branch.down_state == 'u'):
            unknown_branch.append(new_branch)

    return survival_known_branch, unknown_branch



def compute_total_probability(branches):
    """
    Compute the total probability from a list of branches.

    Parameters:
    - branches: List of branches, each containing a probability value.

    Returns:
    - total_probability: Sum of probabilities from all branches.
    """
    
    total_probability = sum(branch.p for branch in branches)
    
    return total_probability



def survivalprob_xi0_brc100(brc_branches, probs, target_xi):
    """
    Step 3: Compute P(S=1, X_i=0) for a specific component using known-branches.

    Parameters:
    - brc_branches: List of branches with known survival status (B_s).
    - probs: Dictionary containing component failure/survival probabilities.
    - target_xi: The specific component X_i to compute P(S=1, X_i=0) for.

    Returns:
    - survival_prob_xi0: Computed probability P(S=1, X_i=0) for the given X_i.
    """

    target_prob2 = 0  # Store P(S=1, X_i=0)

    for branch in brc_branches:
        l_states = branch.down  # Lower bound states
        u_states = branch.up    # Upper bound states
        p_branch = branch.p     # Branch probability

        # Case 1: up_state = 0 (X_i = 0)
        if u_states.get(target_xi, 1) == 0:
            target_prob2 += p_branch  

        # Case 2: down_state = 0 and up_state = 1 (X_i initially fails but later survives)
        elif l_states.get(target_xi, 1) == 0 and u_states.get(target_xi, 1) == 1:
            term = p_branch * probs[target_xi][0]  # P(branch) * P(X_i=0)
            target_prob2 += term

    return target_prob2



def sys_fun_rs(sample):
    val, sys_st, _ = sys_fun(sample)
    return val, sys_st



def run_mcs_for_unknown_branch(brs_u, unknown_branch, probs, sys_fun_rs, cov_t, sys_st_monitor, survival_prob, rand_seed=None):
    """
    Perform Monte Carlo simulation (MCS) for the unknown branch.
    Bayesian inference is used to estimate P(X_i=0, S=1).

    Parameters:
    - unknown_branch: List of unknown branches.
    - probs: Component failure/survival probabilities.
    - sys_fun_rs: Wrapped system function for evaluation.
    - cov_t: Convergence threshold (default: 0.01).
    - rand_seed: Random seed for reproducibility.

    Returns:
    - mcs_result: Dictionary containing MCS results including estimated probability and confidence interval.
    """
    if rand_seed:
        np.random.seed(rand_seed)

    # Extract branch probabilities
    brs_u_probs = [b.p for b in unknown_branch]
    brs_u_prob = round(sum(brs_u_probs), 20)  # Total probability of unknown branches
    
    # Initialize variables for MCS
    samples = []
    samples_sys = np.empty((0, 1), dtype=int)
    sample_probs = []
    nsamp, nmonitoring = 0, 0
    pf, cov = 0.0, 1.0

    while cov > cov_t:
        nsamp += 1

        sample1 = {}
        s_prob1 = {}

        # Select a branch based on probability distribution
        p=[]
        for i in brs_u_probs:
            p.append(i / brs_u_prob)
            
        br_id = np.random.choice(range(len(unknown_branch)), p=p)
        br = brs_u[br_id]

        # Sample each component state
        for e in br.down.keys():
            d = br.down[e]
            u = br.up[e]

            if d < u: # (fail, surv)
                st = np.random.choice(range(d, u + 1), p=[probs[e][d], probs[e][u]])
            else:
                st = d
                       
            sample1[e] = st
            s_prob1[e] = probs[e][st]

        # Run system function
        val, sys_st = sys_fun_rs(sample1)

        if sys_st == 's':  # Survival case
            sys_st = 1
        else:
            sys_st = 0

        samples.append(sample1)
        sample_probs.append(s_prob1)
        samples_sys = np.vstack((samples_sys, [sys_st]))

        if sys_st == sys_st_monitor:
            nmonitoring += 1

        # Bayesian estimation of probability
        if nsamp > 9:
            prior = 0.01
            a, b = prior + nmonitoring, prior + (nsamp - nmonitoring)  # Beta distribution parameters

            p_s = a / (a + b)  # Bayesian probability estimate
            var_s = a * b / (a + b) ** 2 / (a + b + 1)
            std_s = np.sqrt(var_s)

            ps = survival_prob + brs_u_prob * p_s
            std = brs_u_prob * std_s
            cov = std / ps
            
            # Compute confidence interval using beta distribution
            conf_p = 0.95
            low = beta.ppf(0.5 * (1 - conf_p), a, b)
            up = beta.ppf(1 - 0.5 * (1 - conf_p), a, b)
            cint = survival_prob + brs_u_prob * np.array([low, up])

        # Display progress every 1000 samples
        if nsamp % 1000 == 0:
            print(f"nsamp: {nsamp}, P(S=1): {ps:.4e}, COV: {cov:.4e}")
          
    # Store results
    mcs_result = {
        'ps': ps,
        'cov': cov,
        'nsamp': nsamp,
        'cint_low': cint[0],
        'cint_up': cint[1],
    }

    return mcs_result



def eventspace_x1_filter(brs_u, X_n, P_Xn_1, sys_fun):
    """
    Step 1: Event space filtering and system function application to classify survival/unknown branches.

    Parameters:
    - brs_u: List of unknown branches from BRC output.
    - X_n: Target component to filter (e.g., 'e5' for component 5).
    - P_Xn_1: Probability that X_n = 1.
    - sys_fun: System function to determine system state ('s', 'f', or 'u').

    Returns:
    - survival_known_branch: List of branches confirmed as survival.
    - unknown_branch: List of branches classified as unknown.
    """

    lower1_upper1 = []  # Branches where both lower and upper states are 1
    lower0_upper1 = []  # Branches where lower state is 0 and upper state is 1
    lower0_upper1_filtered = []  # Modified branches where lower state is set to 1

    for branch in brs_u:
        lower_state = branch.down.get(X_n, None)  # Lower state of X_n
        upper_state = branch.up.get(X_n, None)    # Upper state of X_n

        # Case 1: Both lower and upper states are 1
        if lower_state == 1 and upper_state == 1:
            lower1_upper1.append(branch)

        # Case 2: Lower state is 0 and upper state is 1
        elif lower_state == 0 and upper_state == 1:
            lower0_upper1.append(branch)

            # Modify the branch by setting lower state to 1 and updating probability
            new_branch = copy.deepcopy(branch)
            new_branch.down[X_n] = 1  # Set lower state to 1
            new_branch.p *= P_Xn_1  # Update probability
            lower0_upper1_filtered.append(new_branch)

    # Apply system function to determine system states
    lower0_upper1_filtered_aftersystemfcn = []
    survival_known_branch = []  # List of confirmed survival branches
    failure_known_branch = []  # List of confirmed failure branches
    unknown_branch = lower1_upper1.copy()  # Initialize unknown branches with lower1_upper1 cases

    for branch in lower0_upper1_filtered:
        # Get component states for system function evaluation
        comps_st_lower = branch.down  
        comps_st_upper = branch.up    

        # Evaluate system function for lower and upper states
        _, sys_st_lower, _ = sys_fun(comps_st_lower)  
        _, sys_st_upper, _ = sys_fun(comps_st_upper)  

        # Copy branch and assign system states
        new_branch = copy.deepcopy(branch)
        new_branch.down_state = sys_st_lower  
        new_branch.up_state = sys_st_upper  

        # If upper state is 's' and lower state is 'f', classify as unknown
        if sys_st_upper == 's' and sys_st_lower == 'f':  
            new_branch.up_state = 'u'  
            new_branch.down_state = 'u'  

        lower0_upper1_filtered_aftersystemfcn.append(new_branch)

        # Add to survival list if both states indicate survival
        if sys_st_upper == 's' and sys_st_lower == 's':
            survival_known_branch.append(new_branch)

        # Add to unknown list if system states are unknown or were modified to unknown
        if (sys_st_upper == 'u' and sys_st_lower == 'u') or (new_branch.up_state == 'u' and new_branch.down_state == 'u'):
            unknown_branch.append(new_branch)
    
    return survival_known_branch, unknown_branch



def survivalprob_xi1_brc100(brc_branches, probs, target_xi):
    """
    Step 3: Compute P(S=1, X_i=1) for a specific component using known-branches.

    Parameters:
    - brc_branches: List of branches with known survival status (B_s).
    - probs: Dictionary containing component failure/survival probabilities.
    - target_xi: The specific component X_i to compute P(S=1, X_i=1) for.

    Returns:
    - survival_prob_xi1: Computed probability P(S=1, X_i=1) for the given X_i.
    """

    target_prob1 = 0  # Store P(S=1, X_i=1)

    for branch in brc_branches:
        l_states = branch.down  # Lower bound states
        u_states = branch.up    # Upper bound states
        p_branch = branch.p     # Branch probability

        # Case 1: down_state = 1 (X_i = 1)
        if l_states.get(target_xi, 1) == 1:
            target_prob1 += p_branch  

        # Case 2: down_state = 0 and up_state = 1 (X_i initially fails but later survives)
        elif l_states.get(target_xi, 1) == 0 and u_states.get(target_xi, 1) == 1:
            term = p_branch * probs[target_xi][1]  # P(branch) * P(X_i=1)
            target_prob1 += term

    return target_prob1



def beta_parameters(mu, cov):
    """
    Computes alpha and beta parameters for a Beta distribution given mean (mu) and coefficient of variation (cov).
    
    Parameters:
    - mu: Mean of the Beta distribution
    - cov: Coefficient of variation (CoV = sigma/mu)
    
    Returns:
    - alpha: Shape parameter alpha
    - beta: Shape parameter beta
    """
    sigma = cov * mu  # Compute standard deviation from CoV
    S = (mu * (1 - mu)) / (sigma ** 2) - 1  # Compute sum α + β

    if S <= 0:
        raise ValueError("Invalid input values: variance is too high for a Beta distribution.")

    alpha = S * mu
    beta = S * (1 - mu)

    return alpha, beta



mp.dps = 50  # Set decimal precision (default 50 digits)

def beta_pdf_mpmath(x, alpha, beta):
    """
    Beta distribution PDF using mpmath
    """
    if x < 0 or x > 1:
        return 0
    return (mp.gamma(alpha + beta) / (mp.gamma(alpha) * mp.gamma(beta))) * (x**(alpha - 1)) * ((1 - x)**(beta - 1))



def f_Y_mpmath(y, alpha1, beta1, alpha2, beta2):
    """
    Computes the probability density function of Y = P1 - P2 using mpmath.
    """
    if y < -1 or y > 1:
        return 0  # P1 - P2 is always within [-1,1]
    
    def integrand(p2):
        return beta_pdf_mpmath(y + p2, alpha1, beta1) * beta_pdf_mpmath(p2, alpha2, beta2)

    # Use mpmath.quad for high-precision integration
    result = mp.quad(integrand, [0, 1])
    return float(result)  # Convert mpmath output to float