import warnings
import scipy.optimize
import numpy as np
import statsmodels.tools.numdiff


class OptimizeError(Exception):
    pass


class ExpectationMaximizationEstimator(object):

    def __init__(self, num_em_iter=100, likelihood_tol=1e-3):
        """Create an expectation maximization estimator

        KwArgs:
            num_em_iter (int): number of em iterations
            likelihood_tol (float): likelihood increase tolerance

        """

        self.num_em_iter = num_em_iter
        self.likelihood_tol = likelihood_tol
        self.likelihood_error_tol = 1e-2
        self.lower_bound_error_tol = 0.5

        self.em_iter = None

    def evaluate_q(self, value, model, states, weights, params, idxs):
        """ Evaluate q function, expected value of complete data log likelihood
        with respect to conditional, given specific parameter value
        
        Args:
            value (numpy.array): parameter value
            model (object): probabilistic model to optimize
            states (numpy.array): variable state matrix
            weights (numpy.array): state weights matrix
            params (list): name of parameter of model to optimize
            idxs (list): list of start indices of params

        Returns:
            numpy.array: expected value of complete data log likelihood

        """

        for p, idx in zip(params, idxs):
            p.value = value[idx:idx+p.length]

        q_value = 0.0
        
        for s, w in zip(states, weights):
            log_likelihood = w * model.log_likelihood(s)
            log_likelihood[w == 0] = 0

            for n in zip(*np.where(np.isnan(log_likelihood))):
                raise OptimizeError('ll is nan', value=value, state=s[n], resp=w[n])

            q_value += np.sum(log_likelihood)

        return -q_value

    def evaluate_q_derivative(self, value, model, states, weights, params, idxs):
        """ Evaluate derivative of q function, expected complete data
        with respect to conditional, given specific parameter value
        
        Args:
            value (numpy.array): parameter value
            model (object): probabilistic model to optimize
            states (numpy.array): variable state matrix
            weights (numpy.array): state weights matrix
            params (list): list of parameters of model to optimize
            idxs (list): list of start indices of params

        Returns:
            numpy.array: partial derivative of expected value of complete data log likelihood

        """

        for p, idx in zip(params, idxs):
            p.value = value[idx:idx+p.length]
        
        q_derivative = np.zeros(value.shape)

        for s, w in zip(states, weights):
            for p, idx in zip(params, idxs):
                log_likelihood_partial = (w[:, np.newaxis] * p.log_likelihood_partial(s))

                for n in zip(*np.where(np.isnan(log_likelihood_partial))):
                    raise OptimizeError('ll partial is nan', value=value, state=s[n], resp=w[n])

                q_derivative[idx:idx+p.length] += np.sum(log_likelihood_partial, axis=0)

        return -q_derivative

    def expectation_step(self, model):
        """ Expectation Step: Calculate weights for variable states
        
        Args:
            model (object): probabilistic model
        
        Returns:
            numpy.array: log likelihood
            numpy.array: variable state matrix
            numpy.array: state weights matrix

        States matrix has shape (S,N,...) for N variables with S states
        Weights matrix has shape (S,N) for N variables with S states

        Weights are interpreted as posterior marginal probabilities.

        """

        log_likelihood, states, weights = model.posterior_marginals()

        return log_likelihood, states, weights

    def maximization_step(self, model, states, weights, params):
        """ Maximization Step.  Maximize Q with respect to a parameter.

        Args:
            model (object): probabilistic model to optimize
            states (numpy.array): variable state matrix
            weights (numpy.array): state weights matrix
            params (list): list of parameters of model to optimize

        Returns:
            numpy.array: optimal parameter maximizing Q

        """

        bounds = np.concatenate([p.bounds for p in params])
        value = np.concatenate([p.value for p in params])
        idxs = np.array([p.length for p in params]).cumsum() - np.array([p.length for p in params])

        print 'lower bound:', -self.evaluate_q(value, model, states, weights, params, idxs)
        q_derivative = -self.evaluate_q_derivative(value, model, states, weights, params, idxs)
        print 'lower bound derivatives:'
        for p, idx in zip(params, idxs):
            print ' ', p.name, q_derivative[idx:idx+p.length]

        result = scipy.optimize.minimize(
            self.evaluate_q,
            value,
            method='L-BFGS-B',
            jac=self.evaluate_q_derivative,
            args=(model, states, weights, params, idxs),
            bounds=bounds,
            options={'ftol':1e-3},
        )

        if not result.success:
            value = np.concatenate([p.value for p in params])
            print 'parameter values: ', value
            print 'analytic derivative: ', self.evaluate_q_derivative(value, model, states, weights, params, idxs)
            print 'numerical derivative: ', statsmodels.tools.numdiff.approx_fprime_cs(value, self.evaluate_q, args=(model, states, weights, params, idxs))
            raise OptimizeError(repr(result))

        for p, idx in zip(params, idxs):
            p.value = result.x[idx:idx+p.length]

        q_value = -result.fun

        print 'new lower bound:', q_value

        return q_value

    def learn_param(self, model, *params):
        """ Optimize h given an initial estimate.

        Args:
            model (object): probabilistic model to optimize
            params (list): list of parameters of model to optimize

        Returns:
            numpy.array: parameter value
            float: log likelihood
            bool: converged

        """

        self.converged = False
        self.error_message = 'no errors'

        log_likelihood_prev = None
        q_value_prev = None

        for self.em_iter in xrange(self.num_em_iter):
            print 'iteration:', self.em_iter

            # Maximize Log likelihood with respect to copy number
            log_likelihood, states, weights = self.expectation_step(model)

            print 'log likelihood:', log_likelihood
            if log_likelihood_prev is not None:
                print 'log likelihood diff:', log_likelihood - log_likelihood_prev

            if log_likelihood_prev is not None and (log_likelihood_prev - log_likelihood) > self.likelihood_error_tol:
                self.error_message = 'log likelihood decreased from {} to {} for e step'.format(log_likelihood_prev, log_likelihood)
                return log_likelihood

            # Maximize Log likelihood with respect to haploid read depth
            try:
                q_value = self.maximization_step(model, states, weights, params)
            except OptimizeError as e:
                self.error_message = 'error during m step: ' + str(e)
                return log_likelihood

            print 'parameter values:'
            for p in params:
                print ' ', p.name, p.value
            print

            if q_value_prev is not None and (q_value_prev - q_value) > self.lower_bound_error_tol:
                warnings.warn('lower bound decreased from {} to {} for m step'.format(q_value_prev, q_value))

            if log_likelihood_prev is not None and abs(log_likelihood_prev - log_likelihood) < self.likelihood_tol:
                self.converged = True
                return log_likelihood

            log_likelihood_prev = log_likelihood
            q_value_prev = q_value

        return log_likelihood


class HardAssignmentEstimator(ExpectationMaximizationEstimator):

    def __init__(self, num_em_iter=100, likelihood_tol=1e-3):
        """ Hard assignment EM

        KwArgs:
            num_em_iter (int): number of em iterations
            likelihood_tol (float): likelihood increase tolerance

        """

        ExpectationMaximizationEstimator.__init__(self, num_em_iter=num_em_iter, likelihood_tol=likelihood_tol)

        self.state = None

    def expectation_step(self, model):
        """ Override expectation step for hard assignment.
        
        Args:
            model (object): probabilistic model
        
        Returns:
            numpy.array: log likelihood
            numpy.array: variable state matrix
            numpy.array: state weights matrix

        States matrix has shape (1,N,...) for N variables
        Weights matrix has shape (1,N) for N variables

        """

        log_likelihood, self.state = self.assignment_step(model)

        S = 1
        N = self.state.shape[0]

        states = self.state[np.newaxis]
        weights = np.ones((S, N))

        return log_likelihood, states, weights

    def assignment_step(self, model):
        """ Assignment Step: Calculate optimal state assignment of variables
        
        Args:
            model (object): probabilistic model
        
        Returns:
            numpy.array: log likelihood
            numpy.array: variable state

        State array has shape (N,...) for N variables

        """

        log_likelihood, state = model.optimal_state()

        return log_likelihood, state


