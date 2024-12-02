import math
import numpy as np
import warnings
from scipy.stats import norm, truncnorm, expon
from typing import List, Tuple, Union
from . import rdp as gaussian_analysis
from decimal import Decimal
from scipy.special import binom

def _compute_rdp(args, order, sample_rate, num_steps):
  #print(_compute_rdp)
  return convert_mgf_rdp(args, order, sample_rate, num_steps)

def convert_mgf_rdp(args, order, sample_rate, num_steps):
    moment = order - 1
    #moment=args['lam']
    clip = args["max_grad_norm"]
    #print(maf(args, moment)/(moment))
    return maf(args, moment, sample_rate, num_steps)/(moment)
#    try:
#        MGF1_1 = ((1-(args['a1']*(order-1)*args['theta']))**(-args['k']))  # Gamma
#        MGF1_3 = (args['lam']/(args['lam']-args['a3']*(order-1)))  # Exponential
#        MGF1_4 = ((np.exp(args['a4']*(order-1)*args['b'])-np.exp(args['a4']*(order-1)*args['a']))/(args['a4']*(order-1)*(args['b']-args['a'])))  # Uniform
#        MGF1 = MGF1_1 * MGF1_3 * MGF1_4
        
#        MGF2_1 = ((1-args['a1']*(-order)*args['theta'])**(-args['k']))  # Gamma
#        MGF2_3 = (args['lam']/(args['lam']-args['a3']*(-order)))  # Exponential
#        MGF2_4 = ((np.exp(args['a4']*(-order)*args['b'])-np.exp(args['a4']*(-order)*args['a']))/(args['a4']*(-order)*(args['b']-args['a'])))  # Uniform
#        MGF2 = MGF2_1 * MGF2_3 * MGF2_4
        
        
#        rdp_lmo_ = (1/(order-1)) * np.log((order*MGF1+(order-1)*MGF2)/(2*order-1))
        
 #   except:
 #       return math.inf
 #   return rdp_lmo_
    
def maf(args, moment, sample_rate, num_steps):
    #moment = args["moment"]
    #moment = args['lam']
    epsilon = args["epsilon"]
    clip = args["max_grad_norm"]
    #print(M_p(args, moment*clip))
    #return math.log(1 +Decimal(sample_rate**2)*(M_p(args, moment*clip, num_steps)-1))
    M = 0
    for k in range(0, moment+1):
        b = binom(moment, k)
        term = b*((1-sample_rate)**(moment-k))*(sample_rate**k)*M_p(args, k*clip, num_steps)
        M=M+term
    
    return num_steps*math.log(M)
    
    numer = (moment+1)*M_p(args, moment)+(moment*M_p(args, -1*(moment+1)))
    denom = ((2*moment)+1)#*math.exp(moment*epsilon)
    try:
        ret = math.log(numer/denom)
    except:
        ret = math.inf
    return ret
    
def M_u():
    pass

def M_p(args, moment, T):
    a1 = args['a1']
    a3 = args['a3']
    a4 = args['a4']
    theta = args["theta"]
    k = args['k']
    mu = args['mu']
    sigma = args['sigma']
    a = args['a']
    b = args['b']
    l = args['l']
    u = args['u']
    lam = args['lam']
    res =1
    if args['gamma']:
        res * mgf_gamma(moment, theta, k)
    if args['uniform']:
        res * mgf_uniform(moment, a, b)
        
    #print(mgf_truncated_normal(l, u, mu, sigma, moment))
    #return mgf_truncated_normal(l, u, mu, sigma, moment)*(mgf_gamma(moment*a1, theta, k))*mgf_uniform(moment*a4, a, b)
    #return ((mgf_truncated_normal(l, u, mu, sigma, moment)**args['truncnorm'])*(mgf_gamma(moment, theta, k)**args['gamma']))*(mgf_uniform(moment, a, b)**args['uniform']))
    #print((mgf_gamma(moment, theta, k)**args['gamma'])*(mgf_uniform(moment, a, b)**args['uniform']))
    return res
    
def mgf_gamma(moment, theta, k):
    if moment >= 1/theta:
        return math.inf
        raise Exception("moment must be less than 1/theta")
    return pow(1-moment*(theta), -1*k)

def mgf_truncated_normal(l, u, mu, sigma, t):
    alpha = (l - mu) / sigma
    beta = (u - mu) / sigma
    num = truncnorm.cdf(beta - sigma * t, alpha, beta, loc=mu, scale=sigma) - truncnorm.cdf(alpha - sigma * t, alpha, beta, loc=mu, scale=sigma)
    den = truncnorm.cdf(beta, alpha, beta, loc=mu, scale=sigma) - truncnorm.cdf(alpha, alpha, beta, loc=mu, scale=sigma)
    return np.exp((mu * t + 0.5 * sigma ** 2 * t ** 2)/2) * num / den
    
def mgf_expon(moment, lam):
    paren = 1-(moment*(1/lam))
    if paren == 0:
        return math.inf
    return 1/paren
    
    
def mgf_uniform(moment, a, b):
    n = math.exp(moment*b) - math.exp(moment*a)
    d = moment*(b-a)
    return n/d
    
def compute_rdp(args, num_steps, orders: Union[List[float], float]
) -> Union[List[float], float]:
    r"""Computes Renyi Differential Privacy (RDP) guarantees of the
    Sampled Gaussian Mechanism (SGM) iterated ``steps`` times.

    Args:
        q: Sampling rate of SGM.
        noise_multiplier: The ratio of the standard deviation of the
            additive Gaussian noise to the L2-sensitivity of the function
            to which it is added. Note that this is same as the standard
            deviation of the additive Gaussian noise when the L2-sensitivity
            of the function is 1.
        steps: The number of iterations of the mechanism.
        orders: An array (or a scalar) of RDP orders.

    Returns:
        The RDP guarantees at all orders; can be ``np.inf``.
    """
    #alpha, rdp = _compute_rdp(args)
    #print(num_steps)
    if isinstance(orders, float):
        rdp = _compute_rdp(args, orders, 1, num_steps)
    else:
        rdp = np.array([_compute_rdp(args, order, 1, num_steps) for order in orders])
    return rdp
    
def compute_rdp_subsample(args, num_steps, delta, orders: Union[List[float], float], sample_rate
) -> Union[List[float], float]:
    if isinstance(orders, float):
        rdp = maf(args, orders, sample_rate, num_steps)
    else:
        rdp = np.array([maf(args, order, sample_rate, num_steps) for order in orders])
    
    #if isinstance(orders, float):
    #    rdp = _compute_rdp(args, orders, sample_rate, num_steps)
    #else:
    #    rdp = np.array([_compute_rdp(args, order, sample_rate, num_steps) for order in orders])
    
    rens = []
    #print(len(rdp))
#    eps, best_alpha = get_privacy_spent(
#        orders=orders, rdp=rdp, delta=delta
#        )
        #sigma = np.sqrt((2*np.log(1.25/delta))**2)/(float(eps)**2)
        #rens.append(gaussian_analysis._compute_rdp(sample_rate, sigma, best_alpha))
        
    np.array(rens)*num_steps
    return np.array(rdp)
        
    


def get_privacy_spent(
    *, orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
) -> Tuple[float, float]:
    r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values atS
    multiple RDP orders and target ``delta``.
    The computation of epslion, i.e. conversion from RDP to (eps, delta)-DP,
    is based on the theorem presented in the following work:
    Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
    International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
    Particullary, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.
    Args:
        orders: An array (or a scalar) of orders (alphas).
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.
    Returns:
        Pair of epsilon and optimal order alpha.
    Raises:
        ValueError
            If the lengths of ``orders`` and ``rdp`` are not equal.
    """
    #orders = [orders] * len(rdp)
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)
    #print(rdp_vec)
    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    eps = (
        rdp_vec
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )
    #print(eps)
    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    if idx_opt == 0 or idx_opt == len(eps) - 1:
        extreme = "smallest" if idx_opt == 0 else "largest"
        warnings.warn(
            f"Optimal order is the {extreme} alpha. Please consider expanding the range of alphas to get a tighter privacy bound."
        )
    return eps[idx_opt], orders_vec[idx_opt]


