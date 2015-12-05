import numpy as np
from scipy.special import gammaln
from scipy.special import digamma

import remixt.likelihood


class ReadCountLikelihood(remixt.likelihood.ReadCountLikelihood):

    def expected_read_count_unopt(self, l, cn):
        """ Unoptimized version of remix.likelihod.expected_read_count
        """

        N = cn.shape[0]
        M = cn.shape[1]
        K = (2, 3)[self.total_cn]

        h = self.h
        p = remixt.likelihood.proportion_measureable_matrix(self.phi, total_cn=self.total_cn)

        x = np.zeros((N, K))
        
        for n in xrange(N):
            
            gamma = np.zeros((2,))
            
            for m in xrange(M):
                
                for ell in xrange(2):
                    
                    gamma[ell] += h[m] * cn[n,m,ell]
            
            x[n,0] = l[n] * p[n,0] * gamma[0]
            x[n,1] = l[n] * p[n,1] * gamma[1]

            if self.total_cn:
                x[n,2] = l[n] * p[n,2] * (gamma[0] + gamma[1])
            
        x += 1e-16

        for n, ell in zip(*np.where(x <= 0)):
            raise ProbabilityError('mu <= 0', n=n, cn=cn[n], l=l[n], h=h, p=p[n], mu=x[n])

        for n, ell in zip(*np.where(np.isnan(x))):
            raise ProbabilityError('mu is nan', n=n, cn=cn[n], l=l[n], h=h, p=p[n], mu=x[n])

        return x


    def _log_likelihood_partial_h_unopt(self, x, l, cn):
        """ Unoptimized version of _log_likelihood_partial_h
        """

        partial_mu = self._log_likelihood_partial_mu_unopt(x, l, cn)
        
        p = remixt.likelihood.proportion_measureable_matrix(self.phi, total_cn=self.total_cn)
        q = self.allele_measurement_matrix()
        
        N = x.shape[0]
        M = cn.shape[1]
        K = x.shape[1]

        partial_h = np.zeros((N, M))
        
        for n in xrange(N):

            for m in xrange(M):
            
                for k in range(K):

                    for ell in xrange(2):
                        
                        partial_h[n,m] += partial_mu[n,k] * cn[n,m,ell] * q[ell,k] * p[n,k] * l[n]
                    
        return partial_h


    def _log_likelihood_partial_phi_unopt(self, x, l, cn):
        """ Unoptimized version of _log_likelihood_partial_phi
        """

        partial_mu = self._log_likelihood_partial_mu_unopt(x, l, cn)
        
        h = self.h
        p = remixt.likelihood.proportion_measureable_matrix(self.phi, total_cn=self.total_cn)
        q = self.allele_measurement_matrix()
        
        N = x.shape[0]
        M = cn.shape[1]
        K = x.shape[1]

        partial_phi = np.zeros((N,))
        
        for n in xrange(N):

            for m in xrange(M):
            
                for k in range(K):

                    for ell in xrange(2):

                        if k == ell and k < 2:
                            partial_phi[n] += partial_mu[n,k] * cn[n,m,ell] * l[n] * h[m]
                    
        return partial_phi



class PoissonDistribution(remixt.likelihood.PoissonDistribution):

    def log_likelihood_unopt(self, x, mu):
        """ Unoptimized version of log_likelihood
        """
        
        N = x.shape[0]
        
        ll = np.zeros((N,))

        for n in xrange(N):
            
            ll[n] += x[n] * np.log(mu[n]) - mu[n] - gammaln(x[n] + 1)
                
        return ll


    def log_likelihood_partial_mu_unopt(self, x, mu):
        """ Unoptimized version of log_likelihood_partial_mu
        """
        
        N = x.shape[0]

        partial_mu = np.zeros((N,))
        
        for n in xrange(N):

            partial_mu[n] = x[n] / mu[n] - 1.
                
        return partial_mu



class PoissonLikelihood(ReadCountLikelihood,remixt.likelihood.PoissonLikelihood):

    def __init__(self, **kwargs):
        """ Poisson read count likelihood model.
        """

        super(PoissonLikelihood, self).__init__(**kwargs)

        self.poisson = PoissonDistribution()


    def _log_likelihood_unopt(self, x, l, cn):
        """ Unoptimized version of _log_likelihood
        """

        N = x.shape[0]
        K = x.shape[1]

        mu = self.expected_read_count(l, cn)

        ll = np.zeros((N,))
        for k in xrange(K):
            ll = ll + self.poisson.log_likelihood_unopt(x[:,k], mu[:,k])

        return ll


    def _log_likelihood_partial_mu_unopt(self, x, l, cn):
        """ Unoptimized version of _log_likelihood_partial_mu
        """

        N = x.shape[0]
        K = x.shape[1]

        mu = self.expected_read_count(l, cn)

        partial_mu = np.zeros((N, K))
        for k in xrange(K):
            partial_mu[:,k] = self.poisson.log_likelihood_partial_mu_unopt(x[:,k], mu[:,k])
        
        return partial_mu



class NegBinDistribution(remixt.likelihood.NegBinDistribution):

    def log_likelihood_unopt(self, x, mu):
        """ Unoptimized version of log_likelihood
        """
        
        N = x.shape[0]

        ll = np.zeros((N,))

        for n in xrange(N):
            
            nb_p = mu[n] / (self.r + mu[n])
            
            ll[n] += gammaln(x[n] + self.r) - gammaln(x[n] + 1) - gammaln(self.r)
            ll[n] += x[n] * np.log(nb_p) + self.r * np.log(1 - nb_p)
                
        return ll


    def log_likelihood_partial_mu_unopt(self, x, mu):
        """ Unoptimized version of log_likelihood_partial_mu
        """
        
        N = x.shape[0]

        partial_mu = np.zeros((N,))

        for n in xrange(N):

            partial_mu[n] = x[n] / mu[n] - (self.r + x[n]) / (self.r + mu[n])

        return partial_mu


    def log_likelihood_partial_r_unopt(self, x, mu):
        """ Unoptimized version of log_likelihood_partial_r
        """

        r = self.r
        
        N = x.shape[0]

        partial_r = np.zeros((N,))

        for n in xrange(N):

            partial_r[n] = (digamma(r + x[n]) - digamma(r) + np.log(r) + 1.
                - np.log(r + mu[n]) - r / (r + mu[n])
                - x[n] / (r + mu[n]))

        return partial_r



class NegBinLikelihood(ReadCountLikelihood,remixt.likelihood.NegBinLikelihood):

    def __init__(self, **kwargs):
        """ Negative binomial read count likelihood model.

        Attributes:
            r (numpy.array): negative binomial read count over-dispersion

        """

        super(NegBinLikelihood, self).__init__(**kwargs)

        self.negbin = [NegBinDistribution(), NegBinDistribution(), NegBinDistribution()]


    def _log_likelihood_unopt(self, x, l, cn):
        """ Unoptimized version of log_likelihood_partial_r
        """
        
        N = x.shape[0]
        K = x.shape[1]

        mu = self.expected_read_count(l, cn)

        ll = np.zeros((N,))
        for k in xrange(K):
            ll = ll + self.negbin[k].log_likelihood_unopt(x[:,k], mu[:,k])
        
        return ll


    def _log_likelihood_partial_mu_unopt(self, x, l, cn):
        """ Unoptimized version of _log_likelihood_partial_mu
        """
        
        N = x.shape[0]
        K = x.shape[1]

        mu = self.expected_read_count(l, cn)

        partial_mu = np.zeros((N, K))
        for k in xrange(K):
            partial_mu[:,k] = self.negbin[k].log_likelihood_partial_mu_unopt(x[:,k], mu[:,k])
        
        return partial_mu


    def _log_likelihood_partial_r_unopt(self, x, l, cn):
        """ Unoptimized version of _log_likelihood_partial_r
        """

        N = x.shape[0]
        K = x.shape[1]

        mu = self.expected_read_count(l, cn)

        partial_r = np.zeros((N, K))
        for k in xrange(K):
            partial_r[:,k] = self.negbin[k].log_likelihood_partial_r_unopt(x[:,k], mu[:,k])
        
        return partial_r





