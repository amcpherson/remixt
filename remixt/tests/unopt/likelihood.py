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



class PoissonLikelihood(remixt.likelihood.PoissonLikelihood, ReadCountLikelihood):

    def _log_likelihood_unopt(self, x, mu):
        """ Unoptimized version of _log_likelihood
        """
        
        N = x.shape[0]
        K = x.shape[1]
        
        ll = np.zeros((N,))

        for n in xrange(N):
            
            for k in xrange(K):
                
                ll[n] += x[n,k] * np.log(mu[n,k]) - mu[n,k] - gammaln(x[n,k] + 1)
                
        return ll


    def _log_likelihood_partial_mu_unopt(self, x, l, cn):
        """ Unoptimized version of _log_likelihood_partial_mu
        """

        mu = self.expected_read_count_unopt(l, cn)
        
        N = x.shape[0]
        K = x.shape[1]

        partial_mu = np.zeros((N, K))
        
        for n in xrange(N):

            for k in range(K):

                partial_mu[n,k] = x[n,k] / mu[n,k] - 1.
                
        return partial_mu



class NegBinLikelihood(remixt.likelihood.NegBinLikelihood, ReadCountLikelihood):

    def _log_likelihood_unopt(self, x, mu):
        """ Unoptimized version of _log_likelihood
        """
        
        N = x.shape[0]
        K = x.shape[1]

        ll = np.zeros((N,))

        for n in xrange(N):
            
            for k in xrange(K):
            
                nb_p = mu[n,k] / (self.r[k] + mu[n,k])
                
                ll[n] += gammaln(x[n,k] + self.r[k]) - gammaln(x[n,k] + 1) - gammaln(self.r[k])
                ll[n] += x[n,k] * np.log(nb_p) + self.r[k] * np.log(1 - nb_p)
                
        return ll


    def _log_likelihood_partial_mu_unopt(self, x, l, cn):
        """ Unoptimized version of _log_likelihood_partial_mu
        """

        mu = self.expected_read_count_unopt(l, cn)
        
        q = np.array([[1, 0, 1], [0, 1, 1]])
        
        N = x.shape[0]
        M = cn.shape[1]
        K = x.shape[1]

        partial_mu = np.zeros((N, K))

        for n in xrange(N):

            for k in range(K):

                partial_mu[n,k] = x[n,k] / mu[n,k] - (self.r[k] + x[n,k]) / (self.r[k] + mu[n,k])

        return partial_mu


    def _log_likelihood_partial_r_unopt(self, x, l, cn):
        """ Unoptimized version of _log_likelihood_partial_r
        """

        mu = self.expected_read_count_unopt(l, cn)

        r = self.r
        
        N = x.shape[0]
        K = x.shape[1]

        partial_r = np.zeros((N, K))

        for n in xrange(N):

            for k in range(K):

                partial_r[n,k] = (digamma(r[k] + x[n,k]) - digamma(r[k]) + np.log(r[k]) + 1.
                    - np.log(r[k] + mu[n,k]) - r[k] / (r[k] + mu[n,k])
                    - x[n,k] / (r[k] + mu[n,k]))

        return partial_r


