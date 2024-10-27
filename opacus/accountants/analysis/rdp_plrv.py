import math
import numpy as np
import warnings
from scipy.stats import norm, truncnorm

def _compute_rdp(args):
  return convert_mgf_rdp(args)

def convert_mgf_rdp(args):
    moment = args["moment"]
    clip = args["max_grad_norm"]
    return moment+1, maf(args)/(moment*clip)
    
def maf(args):
    moment = args["moment"]
    epsilon = args["epsilon"]
    clip = args["max_grad_norm"]
    numer = (moment+1)*M_p(args, moment*clip)+(moment*M_p(args, -1*(moment+1)*clip))
    denom = ((2*moment)+1)#*math.exp(moment*epsilon)
    #print(numer)
    return math.log(numer/denom)
    
def M_u():
    pass

def M_p(args, moment):
    theta = args["theta"]
    k = args['k']
    mu = args['mu']
    sigma = args['sigma']
    a = args['a']
    b = args['b']
    l = args['l']
    u = args['u']
    #print(mgf_truncated_normal(l, u, mu, sigma, moment))
    return mgf_truncated_normal(l, u, mu, sigma, moment)#(mgf_gamma(moment, theta, k))#*mgf_uniform(moment, a, b)
    
def mgf_gamma(moment, theta, k):
    if moment >= 1/theta:
         raise Exception("moment must be less than 1/theta")
    return pow(1-moment*(theta), -1*k)

def mgf_truncated_normal(l, u, mu, sigma, t):
    alpha = (l - mu) / sigma
    beta = (u - mu) / sigma
    num = norm.cdf(beta - sigma * t) - norm.cdf(alpha - sigma * t)
    den = norm.cdf(beta) - norm.cdf(alpha)
    return np.exp((mu * t + 0.5 * sigma ** 2 * t ** 2)/2) * num / den
    
    
def mgf_uniform(moment, a, b):
    n = math.exp(moment*b) - math.exp(moment*a)
    d = moment*(b-a)
    return n/d
    
def compute_rdp(args, num_steps):
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
    alpha, rdp = _compute_rdp(args)
    #print(rdp)
    
    return alpha, rdp * num_steps


def get_privacy_spent(
    *, orders: float, rdp, delta: float
):
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


