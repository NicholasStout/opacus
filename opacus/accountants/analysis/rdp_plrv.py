import math

def _compute_rdp

def convert_mgf_rdp(gamma):
    return gamma+1, mgf(gamma)/gamma
    
def maf(gamma,  epsilon):
    numer = (gamma+1)*M_p(gamma*self.max_grad_norm)#+(gamma*M_p(-1*(gamma+1)*self.max_grad_norm)
    denom = ((2*gamma)+1)*math.exp(gamma*epsilon)
    return math.log10(numer/denom)
    
def M_u():
    pass

def M_p(evaluate):
    return mgf_gamma()+mgf_normal()+mgf_uniform()
    
def mgf_gamma(moment, theta, k):
    return pow(1-moment*(1/theta)), -1*k)

def mgf_normal(moment, mu, sigmaSqr):
    return math.exp((moment*mu)+(0.5*sigmaSqr*pow(moment, 2)))
    
def mgf_uniform(moment, a, b):
    n = math.exp(moment*b) - math.exp(moment*a)
    d = moment*(b-a)
    return n/d
    
def compute_rdp(
    *, q: float, noise_multiplier: float, steps: int, orders: Union[List[float], float]
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
    if isinstance(orders, float):
        rdp = _compute_rdp(q, noise_multiplier, orders)
    else:
        rdp = np.array([_compute_rdp(q, noise_multiplier, order) for order in orders])

    return rdp * steps


def get_privacy_spent(
    *, orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
) -> Tuple[float, float]:
    r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
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


