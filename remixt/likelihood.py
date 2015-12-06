import numpy as np
import pandas as pd
import scipy
import scipy.misc
from scipy.special import gammaln
from scipy.special import betaln
from scipy.special import digamma

import remixt.nb_overdispersion
import remixt.utils



class ProbabilityError(ValueError):
    def __init__(self, message, **variables):
        """ Error calculating a probability.

        Args:
            message (str): message detailing error

        KwArgs:
            **variables: variables to be printed

        """

        for name, value in variables.iteritems():
            message += '\n{0}={1}'.format(name, value)

        ValueError.__init__(self, message)



def estimate_phi(x):
    """ Infer proportion of genotypable reads.

    Args:
        x (numpy.array): major, minor, and total read counts

    Returns:
        numpy.array: estimate of proportion of genotypable reads.

    """

    phi = x[:,0:2].sum(axis=1).astype(float) / (x[:,2].astype(float) + 1.0)

    return phi



def proportion_measureable_matrix(phi, total_cn=True):
    """ Proportion reads measureable matrix.

    Args:
        phi (numpy.array): estimate of proportion of genotypable reads.    

    KwArgs:
        total_cn (boolean): include proportion for total copy number

    Returns:
        numpy.array: N * K dim array, segment to measurement transform

    """

    if total_cn:
        return np.vstack([phi, phi, np.ones(phi.shape)]).T
    else:
        return np.vstack([phi, phi]).T



def calculate_mean_cn(h, x, l):
    """ Infer proportion of genotypable reads.

    Args:
        h (numpy.array): haploid read depths, h[0] for normal
        x (numpy.array): major, minor, and total read counts
        l (numpy.array): segment lengths

    Returns:
        numpy.array: N * L dim array, per segment per allele mean copy number

    """

    phi = estimate_phi(x)
    p = proportion_measureable_matrix(phi, total_cn=True)

    # Avoid divide by zero
    x = x + 1e-16
    l = l + 1e-16
    p = p + 1e-16

    # Calculate the total haploid depth of the tumour clones
    h_t = ((x[:,0:2] / p[:,0:2]).T / l).T

    for n, ell in zip(*np.where(np.isnan(h_t))):
        raise ProbabilityError('h_t is nan', n=n, x=x[n], l=l[n], h=h, p=p[n])

    # Calculate the dominant cn assuming no divergence
    dom_cn = (h_t - h[0]) / h[1:].sum()

    return dom_cn



class ReadCountLikelihood(object):

    def __init__(self, **kwargs):
        """ Abstract read count likelihood model.

        KwArgs:
            min_length_likelihood (int): minimum length for likelihood calculation
            total_cn (boolean): model total copy number

        Attributes:
            h (numpy.array): haploid read depths, h[0] for normal
            phi (numpy.array): proportion genotypable reads

        """

        self.min_length_likelihood = kwargs.get('min_length_likelihood', 10000)
        self.total_cn = kwargs.get('total_cn', True)

        self.param_partial_func = dict()
        self.param_partial_func['h'] = self._log_likelihood_partial_h
        self.param_partial_func['phi'] = self._log_likelihood_partial_phi

        self.param_bounds = dict()
        self.param_bounds['h'] = (0., 10.)
        self.param_bounds['phi'] = (0., 1.)

        self.param_per_segment = dict()
        self.param_per_segment['h'] = False
        self.param_per_segment['phi'] = True

        self.mask = None


    def add_amplification_mask(self, x, l, cn_max):
        """ Add a mask for highly amplified regions.

        Args:
            x (numpy.array): major, minor, and total read counts
            l (numpy.array): segment lengths
            cn_max (int): max unmasked dominant copy number

        Returns:
            float: proportion of the genome masked

        """

        dom_cn = calculate_mean_cn(self.h, x, l)
        dom_cn = np.clip(dom_cn.round().astype(int), 0, int(1e6))

        self.mask = np.all(dom_cn <= cn_max, axis=1)


    @property
    def h(self):
        return self._h
    @h.setter
    def h(self, value):
        self._h = value.copy()
        self._h[self._h < 0.] = 0.


    @property
    def phi(self):
        return self._phi
    @phi.setter
    def phi(self, value):
        self._phi = value.copy()
        self._phi[self._phi < 0.] = 0.


    def estimate_parameters(self, x, l):
        """ Offline parameter inference.

        Args:
            x (numpy.array): major, minor, and total read counts
            l (numpy.array): segment lengths

        """
        
        self.phi = estimate_phi(x)


    def allele_measurement_matrix(self):
        """ Allele measurement matrix.

        Returns:
            numpy.array: L * K dim array, allele to measurement transform

        """

        q = np.array([[1, 0, 1], [0, 1, 1]])

        if not self.total_cn:
            q = q[:,0:2]

        return q


    def expected_read_count(self, l, cn):
        """Calculate expected major, minor and total read counts.
        
        Args:
            l (numpy.array): segment lengths
            cn (numpy.array): copy number state
        
        Returns:
            numpy.array: expected read depths
        """
        
        h = self.h
        p = proportion_measureable_matrix(self.phi, total_cn=self.total_cn)
        q = self.allele_measurement_matrix()

        gamma = np.sum(cn * np.vstack([h, h]).T, axis=-2)

        x1 = np.dot(q.T, gamma.T).T

        x2 = x1 * p
        
        x3 = (x2.T * l.T).T

        x3 += 1e-16

        for n, ell in zip(*np.where(x3 <= 0)):
            raise ProbabilityError('mu <= 0', n=n, cn=cn[n], l=l[n], h=h, p=p[n], mu=x3[n])

        for n, ell in zip(*np.where(np.isnan(x3))):
            raise ProbabilityError('mu is nan', n=n, cn=cn[n], l=l[n], h=h, p=p[n], mu=x3[n])

        return x3


    def log_likelihood(self, x, l, cn):
        """ Evaluate log likelihood
        
        Args:
            x (numpy.array): major, minor, and total read counts
            l (numpy.array): segment lengths
            cn (numpy.array): copy number state

        Returns:
            numpy.array: log likelihood per segment

        """

        if not self.total_cn:
            x = x[:,0:2]

        ll = self._log_likelihood(x, l, cn)

        for n in zip(*np.where(np.isnan(ll))):
            raise ProbabilityError('ll is nan', n=n, x=x[n], cn=cn[n], l=l[n])

        ll[np.where(np.any(cn < 0, axis=(-1, -2)))] = -np.inf

        ll[l < self.min_length_likelihood] = 0.0

        if self.mask is not None:
            ll[~self.mask] = 0.0

        return ll


    def log_likelihood_partial_param(self, x, l, cn, param):
        """ Evaluate partial derivative of log likelihood with respect to a parameter
        
        Args:
            x (numpy.array): major, minor, and total read counts
            l (numpy.array): segment lengths
            cn (numpy.array): copy number state
            param (str): parameter name

        Returns:
            numpy.array: partial derivative of log likelihood per segment per clone

        """

        if not self.total_cn:
            x = x[:,0:2]

        mu = self.expected_read_count(l, cn)

        ll_partial = self.param_partial_func[param](x, l, cn)

        ll_partial[l < self.min_length_likelihood,:] = 0.0

        if self.mask is not None:
            ll_partial[~self.mask,:] = 0.0

        for n in zip(*np.where(np.isnan(ll_partial))):
            raise ProbabilityError('ll derivative is nan', n=n, x=x[n], cn=cn[n], l=l[n], h=h, p=p[n], mu=mu[n])

        return ll_partial


    def _log_likelihood_partial_h(self, x, l, cn):
        """ Evaluate partial derivative of log likelihood with respect to h
        
        Args:
            x (numpy.array): major, minor, and total read counts
            l (numpy.array): segment lengths
            cn (numpy.array): copy number state
        
        Returns:
            numpy.array: log likelihood derivative per segment per clone
            
        The partial derivative of the log likelihood with respect to h[m] is:
        
            sum_k a[n,k] * cn[n,m,ell] * q[ell,k] * p[n,k] * l[n]

        where a[n,k] is the partial derivative of p(x[n,k]|.) with respect to mu[n,k]

        """

        partial_mu = self._log_likelihood_partial_mu(x, l, cn)
        
        p = proportion_measureable_matrix(self.phi, total_cn=self.total_cn)
        q = self.allele_measurement_matrix()

        partial_h = np.einsum('...l,...jk,...kl,...l,...->...j', partial_mu, cn, q, p, l)
        
        return partial_h


    def _log_likelihood_partial_phi(self, x, l, cn):
        """ Evaluate partial derivative of log likelihood with respect to phi
        
        Args:
            x (numpy.array): major, minor, and total read counts
            l (numpy.array): segment lengths
            cn (numpy.array): copy number state
        
        Returns:
            numpy.array: log likelihood derivative per segment per clone
            
        The partial derivative of the log likelihood with respect to phi[n] is:
        
            sum_k a[n,k] * cn[n,m,ell] * l[n] * h[m] * I(k<>3) * I(k=ell)

        where a[n,k] is the partial derivative of p(x[n,k]|.) with respect to mu[n,k]
            
        """

        h = self.h

        partial_mu = self._log_likelihood_partial_mu(x, l, cn)

        partial_phi = (partial_mu[:,0] * l * np.dot(cn[:,:,0], h) + 
            partial_mu[:,1] * l * np.dot(cn[:,:,1], h))

        return partial_phi[:,np.newaxis]



class PoissonDistribution(object):

    def log_likelihood(self, x, mu):
        """ Calculate the poisson read count log likelihood.
        
        Args:
            x (numpy.array): measured read counts
            mu (numpy.array): expected read counts
        
        Returns:
            float: log likelihood
            
        The pmf of the negative binomial is:
        
            mu^x * e^-mu / x!
            
        The log likelihood is thus:
        
            x * log(mu) - mu - log(x!)
        """

        mu[mu <= 0] = 1

        ll = x * np.log(mu) - mu - gammaln(x + 1)

        return ll


    def log_likelihood_partial_mu(self, x, mu):
        """ Calculate the partial derivative of the poisson read count log likelihood
        with respect to mu
        
        Args:
            x (numpy.array): measured read counts
            mu (numpy.array): expected read counts
        
        Returns:
            numpy.array: log likelihood derivative
            
        The partial derivative of the log pmf of the poisson with 
        respect to mu is:
        
            x / mu - 1

        """

        partial_mu = x / mu - 1.
        
        return partial_mu


class PoissonLikelihood(ReadCountLikelihood):

    def __init__(self, **kwargs):
        """ Poisson read count likelihood model.

        """

        self.poisson = PoissonDistribution()

        super(PoissonLikelihood, self).__init__(**kwargs)


    def _log_likelihood(self, x, l, cn):
        """ Calculate the poisson read count log likelihood.
        
        Args:
            x (numpy.array): major, minor, and total read counts
            l (numpy.array): segment lengths
            cn (numpy.array): copy number state
        
        Returns:
            float: log likelihood per segment

        """

        N = x.shape[0]
        K = x.shape[1]

        mu = self.expected_read_count(l, cn)

        ll = np.zeros((N,))
        for k in xrange(K):
            ll = ll + self.poisson.log_likelihood(x[:,k], mu[:,k])

        return ll


    def _log_likelihood_partial_mu(self, x, l, cn):
        """ Calculate the partial derivative of the poisson read count log likelihood
        with respect to mu
        
        Args:
            x (numpy.array): major, minor, and total read counts
            l (numpy.array): segment lengths
            cn (numpy.array): copy number state
        
        Returns:
            numpy.array: log likelihood derivative per segment per measurement
            
        """

        N = x.shape[0]
        K = x.shape[1]

        mu = self.expected_read_count(l, cn)

        partial_mu = np.zeros((N, K))
        for k in xrange(K):
            partial_mu[:,k] = self.poisson.log_likelihood_partial_mu(x[:,k], mu[:,k])
        
        return partial_mu



class NegBinDistribution(object):

    def __init__(self, **kwargs):
        """ Negative binomial read count likelihood model.

        Attributes:
            r (numpy.array): negative binomial read count over-dispersion

        """

        self.r = 100.


    def estimate_parameters(self, x, l):
        """ Offline parameter inference.

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments

        """
        
        super(NegBinLikelihood, self).estimate_parameters(x, l)

        K = (2, 3)[self.total_cn]
        p = proportion_measureable_matrix(self.phi, total_cn=self.total_cn)
        
        self.r = np.array([remixt.nb_overdispersion.infer_disperion(x[:,k], l*p[:,k]) for k in xrange(K)])


    def log_likelihood(self, x, mu):
        """ Calculate negative binomial read count log likelihood.
        
        Args:
            x (numpy.array): measured read counts
            mu (numpy.array): expected read counts
        
        Returns:
            float: log likelihood per segment
            
        The pmf of the negative binomial is:
        
            C(x + r - 1, x) * p^x * (1-p)^r
            
        where p = mu / (r + mu), with mu the mean of the distribution.  The log
        likelihood is thus:
        
            log(G(x+r)) - log(G(x+1)) - log(G(r)) + x * log(p) + r * log(1 - p)
        """
        
        nb_p = mu / (self.r + mu)

        nb_p[nb_p < 0.] = 0.5
        nb_p[nb_p > 1.] = 0.5

        ll = (gammaln(x + self.r) - gammaln(x + 1) - gammaln(self.r)
            + x * np.log(nb_p) + self.r * np.log(1 - nb_p))
        
        return ll


    def log_likelihood_partial_mu(self, x, mu):
        """ Calculate the partial derivative of the negative binomial read count
        log likelihood with respect to mu

        Args:
            x (numpy.array): measured read counts
            mu (numpy.array): expected read counts
        
        Returns:
            numpy.array: log likelihood derivative per segment
            
        The partial derivative of the log pmf of the negative binomial with 
        respect to mu is:
        
            x / mu - (r + x) / (r + mu)

        """
        
        partial_mu = x / mu - (self.r + x) / (self.r + mu)
        
        return partial_mu


    def log_likelihood_partial_r(self, x, mu):
        """ Calculate the partial derivative of the negative binomial read count
        log likelihood with respect to r
        
        Args:
            x (numpy.array): measured read counts
            mu (numpy.array): expected read counts
        
        Returns:
            numpy.array: log likelihood derivative per segment
            
        The partial derivative of the log pmf of the negative binomial with 
        respect to r is:
        
            digamma(r + x) - digamma(r) + log(r) + 1
                - log(r + mu) - r / (r + mu)
                - x / (r + mu)

        """

        r = self.r

        partial_r = (digamma(r + x) - digamma(r) + np.log(r) + 1.
            - np.log(r + mu) - r / (r + mu)
            - x / (r + mu))
        
        return partial_r



class NegBinLikelihood(ReadCountLikelihood):

    def __init__(self, **kwargs):
        """ Negative binomial read count likelihood model.

        Attributes:
            r (numpy.array): negative binomial read count over-dispersion

        """

        super(NegBinLikelihood, self).__init__(**kwargs)

        self.param_partial_func['r'] = self._log_likelihood_partial_r

        self.param_bounds['r'] = (0., np.inf)

        self.param_per_segment['r'] = False

        self.negbin = [NegBinDistribution(), NegBinDistribution(), NegBinDistribution()]


    @property
    def r(self):
        return np.array([nb.r for nb in self.negbin])
    @r.setter
    def r(self, value):
        for idx, val in enumerate(value):
            self.negbin[idx].r = max(0., val)


    def estimate_parameters(self, x, l):
        """ Offline parameter inference.

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments

        """
        
        super(NegBinLikelihood, self).estimate_parameters(x, l)

        K = (2, 3)[self.total_cn]
        p = proportion_measureable_matrix(self.phi, total_cn=self.total_cn)
        
        self.r = np.array([remixt.nb_overdispersion.infer_disperion(x[:,k], l*p[:,k]) for k in xrange(K)])


    def _log_likelihood(self, x, l, cn):
        """ Calculate negative binomial read count log likelihood.
        
        Args:
            x (numpy.array): major, minor, and total read counts
            l (numpy.array): segment lengths
            cn (numpy.array): copy number state
        
        Returns:
            float: log likelihood per segment

        """
        
        N = x.shape[0]
        K = x.shape[1]

        mu = self.expected_read_count(l, cn)

        ll = np.zeros((N,))
        for k in xrange(K):
            ll = ll + self.negbin[k].log_likelihood(x[:,k], mu[:,k])
        
        return ll


    def _log_likelihood_partial_mu(self, x, l, cn):
        """ Calculate the partial derivative of the negative binomial read count
        log likelihood with respect to mu

        Args:
            x (numpy.array): major, minor, and total read counts
            l (numpy.array): segment lengths
            cn (numpy.array): copy number state
        
        Returns:
            numpy.array: log likelihood derivative per segment per measurement

        """
        
        N = x.shape[0]
        K = x.shape[1]

        mu = self.expected_read_count(l, cn)

        partial_mu = np.zeros((N, K))
        for k in xrange(K):
            partial_mu[:,k] = self.negbin[k].log_likelihood_partial_mu(x[:,k], mu[:,k])
        
        return partial_mu


    def _log_likelihood_partial_r(self, x, l, cn):
        """ Calculate the partial derivative of the negative binomial read count
        log likelihood with respect to r
        
        Args:
            x (numpy.array): major, minor, and total read counts
            l (numpy.array): segment lengths
            cn (numpy.array): copy number state
        
        Returns:
            numpy.array: log likelihood derivative per segment per measurement
            
        """

        N = x.shape[0]
        K = x.shape[1]

        mu = self.expected_read_count(l, cn)

        partial_r = np.zeros((N, K))
        for k in xrange(K):
            partial_r[:,k] = self.negbin[k].log_likelihood_partial_r(x[:,k], mu[:,k])
        
        return partial_r



class BetaBinDistribution(object):

    def __init__(self, **kwargs):
        """ Beta binomial allele counts likelihood model.

        Attributes:
            M (numpy.array): beta binomial allele counts over-dispersion

        """

        self.M = 100.


    def log_likelihood(self, x, p):
        """ Calculate beta binomial allele count log likelihood.
        
        Args:
            x (numpy.array): measured major, minor read counts
            p (numpy.array): expected minor allele fraction
        
        Returns:
            float: log likelihood per segment
            
        The pmf of the beta binomial is:
        
            C(n, k) * B(k + M * p, n - k + M * (1 - p)) / B(M * p, M * (1 - p))

        Where p=mu[1]/(mu[0]+mu[1]), k=x[1], n=x[0]+x[1], and M is the over-dispersion
        parameter.

        The log likelihood is thus:
        
            log(G(n+1)) - log(G(k+1)) - log(G(n-k+1))
                + log(G(k + M * p)) + log(G(n - k + M * (1 - p)))
                - log(G(n + M))
                - log(G(M * p)) - log(G(M * (1 - p)))
                + log(G(M))

        """

        k = x[:,1]
        n = x[:,0] + x[:,1]
        M = self.M

        ll = (gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)
            + gammaln(k + M * p) + gammaln(n - k + M * (1 - p))
            - gammaln(n + M)
            - gammaln(M * p) - gammaln(M * (1 - p))
            + gammaln(M))
        
        return ll


    def log_likelihood_partial_p(self, x, p):
        """ Calculate the partial derivative of the beta binomial allele count
        log likelihood with respect to p

        Args:
            x (numpy.array): measured major, minor, and total read counts
            p (numpy.array): expected minor allele fraction
        
        Returns:
            numpy.array: log likelihood derivative per segment per clone

        The log likelihood of the beta binomial in terms of p, and with beta
        functions expanded is:

            log(G(k + M * p))
                + log(G(n - k + M * (1 - p)))
                - log(G(M * p))
                - log(G(M * (1 - p)))

        The partial derivative of the log pmf of the beta binomial with 
        respect to p is:

            M * digamma(k + M * p)
                + (-M) * digamma(n - k + M * (1 - p))
                - M * digamma(M * p)
                - (-M) * digamma(M * (1 - p))

        """

        k = x[:,1]
        n = x[:,0] + x[:,1]
        M = self.M

        partial_p = (M * digamma(k + M * p)
            + (-M) * digamma(n - k + M * (1 - p))
            - M * digamma(M * p)
            - (-M) * digamma(M * (1 - p)))
        
        return partial_p


    def log_likelihood_partial_M(self, x, p):
        """ Calculate the partial derivative of the beta binomial allele count
        log likelihood with respect to p

        Args:
            x (numpy.array): measured major, minor, and total read counts
            p (numpy.array): expected minor allele fraction
        
        Returns:
            numpy.array: log likelihood derivative per segment per clone

        The log likelihood of the beta binomial in terms of M, and with beta
        functions expanded is:

            log(G(k + M * p))
                + log(G(n - k + M * (1 - p)))
                - log(G(n + M))
                - log(G(M * p))
                - log(G(M * (1 - p)))
                + log(G(M))

        The partial derivative of the log pmf of the beta binomial with 
        respect to p is:

            p * digamma(k + M * p)
                + (1 - p) * digamma(n - k + M * (1 - p))
                - digamma(n + M)
                - p * digamma(M * p)
                - (1 - p) * digamma(M * (1 - p))
                + digamma(M)

        """

        k = x[:,1]
        n = x[:,0] + x[:,1]
        M = self.M

        partial_M = (p * digamma(k + M * p)
            + (1 - p) * digamma(n - k + M * (1 - p))
            - digamma(n + M)
            - p * digamma(M * p)
            - (1 - p) * digamma(M * (1 - p))
            + digamma(M))

        return partial_M


