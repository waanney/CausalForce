import torch
import torch.nn.functional as F

_EPS = 1e-8

def _to_var(logvar):
    # logvar → variance and standard deviation (with numerical stabilization)
    var = torch.exp(logvar).clamp_min(_EPS)
    std = torch.sqrt(var)
    return var, std

def expected_sq_l2(mu1, logvar1, mu2, logvar2):
    """
    E[ ||z1 - z2||^2 ] = ||mu1 - mu2||^2 + tr(Sigma1 + Sigma2)
    Shapes are broadcastable; sum over the last dimension and return a scalar per entry.
    """
    var1, _ = _to_var(logvar1)
    var2, _ = _to_var(logvar2)
    mean_term = ((mu1 - mu2) ** 2).sum(dim=-1)
    trace_term = (var1 + var2).sum(dim=-1)
    return mean_term + trace_term

def kl_gaussian_diag(mu1, logvar1, mu2, logvar2):
    """
    KL( N(mu1, S1) || N(mu2, S2) ) for diagonal covariances; closed-form.
    Returns the sum over the last dimension.
    """
    var1, _ = _to_var(logvar1)
    var2, _ = _to_var(logvar2)
    inv_var2 = 1.0 / var2
    d = mu1.shape[-1]
    diff = (mu2 - mu1)

    trace = (inv_var2 * var1).sum(dim=-1)
    quad = (diff * diff * inv_var2).sum(dim=-1)
    logdet = (logvar2 - logvar1).sum(dim=-1)
    return 0.5 * (trace + quad - d + logdet)

def sym_kl_gaussian_diag(mu1, logvar1, mu2, logvar2):
    # Symmetric KL: KL(p||q) + KL(q||p)
    return kl_gaussian_diag(mu1, logvar1, mu2, logvar2) + \
           kl_gaussian_diag(mu2, logvar2, mu1, logvar1)

def bhattacharyya_gaussian_diag(mu1, logvar1, mu2, logvar2):
    """
    Bhattacharyya distance for diagonal Gaussians:
    DB = 1/8 (Δμ)^T Σ^{-1} (Δμ) + 1/2 ln( det(Σ) / sqrt(det Σ1 det Σ2) )
    where Σ = (Σ1 + Σ2)/2.
    """
    var1, _ = _to_var(logvar1)
    var2, _ = _to_var(logvar2)
    diff = mu1 - mu2

    var = 0.5 * (var1 + var2)                      # Σ
    inv_var = 1.0 / var
    term1 = 0.125 * (diff * diff * inv_var).sum(dim=-1)

    # logdet(Σ) - 0.5*(logdet Σ1 + logdet Σ2) for diagonal covariance
    logdet_var = torch.log(var).sum(dim=-1)
    logdet_v1 = logvar1.sum(dim=-1)
    logdet_v2 = logvar2.sum(dim=-1)
    term2 = 0.5 * (logdet_var - 0.5 * (logdet_v1 + logdet_v2))
    return term1 + term2

def wasserstein2_gaussian_diag(mu1, logvar1, mu2, logvar2, squared=True):
    """
    W2^2( N1, N2 ) = ||μ1-μ2||^2 + Tr(Σ1 + Σ2 - 2(Σ1^{1/2} Σ2 Σ1^{1/2})^{1/2})
    For diagonal covariances:
       ||μ1-μ2||^2 + Σ (σ1 + σ2 - 2*sqrt(σ1 σ2))
    where σ denotes the variance (not std).
    """
    var1, _ = _to_var(logvar1)
    var2, _ = _to_var(logvar2)
    mean_term = ((mu1 - mu2) ** 2).sum(dim=-1)
    # For diagonal covariances, the cross term is 2 * sum(sqrt(σ1 σ2))
    cov_term = (var1 + var2 - 2.0 * torch.sqrt(var1 * var2 + _EPS)).sum(dim=-1)
    w2_sq = mean_term + cov_term
    return w2_sq if squared else torch.sqrt(torch.clamp(w2_sq, min=_EPS))

def gaussian_nll_of_sample(z, mu, logvar, reduce_dim=True):
    """
    -log N(z | mu, diag(var)); commonly used as a "compatibility" measure.
    """
    var, _ = _to_var(logvar)
    quad = ((z - mu) ** 2) / var
    nll = 0.5 * (quad + torch.log(var + _EPS)).sum(dim=-1)
    return nll if not reduce_dim else nll
