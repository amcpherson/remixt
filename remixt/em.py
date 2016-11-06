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

    def evaluate_q(self, value, model, weights, param):
        """ Evaluate q function, expected value of complete data log likelihood
        with respect to conditional, given specific parameter value
        
        Args:
            value (numpy.array): parameter value
            model (object): probabilistic model to optimize
            weights (numpy.array): state weights matrix
            param (OptimizeParameter): parameter to optimize

        Returns:
            numpy.array: expected value of complete data log likelihood

        """
        if param.is_scalar and (value < param.bounds[0] or value > param.bounds[1]):
            return np.inf
        
        elif not param.is_scalar:
            for i in range(len(value)):
                if (value[i] < param.bounds[i][0] or value[i] > param.bounds[i][1]):
                    return np.inf

        param.value = value

        q_value = 0.0
        
        for s, w in enumerate(weights):
            log_likelihood = w * model.log_likelihood(s)
            log_likelihood[w == 0] = 0

            for n in zip(*np.where(np.isnan(log_likelihood))):
                raise OptimizeError('ll is nan', value=value, state=s, resp=w[n])

            q_value += np.sum(log_likelihood)

        return -q_value

    def evaluate_q_derivative(self, value, model, weights, param):
        """ Evaluate derivative of q function, expected complete data
        with respect to conditional, given specific parameter value
        
        Args:
            value (numpy.array): parameter value
            model (object): probabilistic model to optimize
            weights (numpy.array): state weights matrix
            param (OptimizeParameter): parameter to optimize

        Returns:
            numpy.array: partial derivative of expected value of complete data log likelihood

        """
        if not param.is_scalar:
            for i in range(len(value)):
                if (value[i] < param.bounds[i][0] or value[i] > param.bounds[i][1]):
                    return np.inf

        param.value = value
        
        q_derivative = np.zeros(value.shape)

        for s, w in enumerate(weights):
            log_likelihood_partial = (w[:, np.newaxis] * param.log_likelihood_partial(s))

            for n in zip(*np.where(np.isnan(log_likelihood_partial))):
                raise OptimizeError('ll partial is nan', value=value, state=s, resp=w[n])

            q_derivative += np.sum(log_likelihood_partial, axis=0)

        return -q_derivative

    def expectation_step(self, model):
        """ Expectation Step: Calculate weights for variable states
        
        Args:
            model (object): probabilistic model
        
        Returns:
            numpy.array: log likelihood
            numpy.array: state weights matrix

        Weights matrix has shape (S,N) for N variables with S states

        Weights are interpreted as posterior marginal probabilities.

        """

        log_likelihood, weights = model.posterior_marginals()

        return log_likelihood, weights

    def maximization_step(self, model, weights, params):
        """ Maximization Step.  Maximize Q with respect to a parameter.

        Args:
            model (object): probabilistic model to optimize
            weights (numpy.array): state weights matrix
            params (list): list of parameters of model to optimize

        Returns:
            numpy.array: optimal parameter maximizing Q

        """
        
        for param in params:
            print 'optimizing:', param.name,
            print 'init:', param.value,
            print 'is_scalar:', param.is_scalar,
            print 'bounds:', param.bounds
            
            if param.is_scalar:
                result = scipy.optimize.brute(
                    self.evaluate_q,
                    args=(model, weights, param),
                    ranges=[param.bounds],
                    full_output=True,
                )

                param.value = result[0]
                q_value = -result[1]

            else:
                result = scipy.optimize.minimize(
                    self.evaluate_q,
                    param.value,
                    method='L-BFGS-B',
                    jac=self.evaluate_q_derivative,
                    args=(model, weights, param),
                    bounds=param.bounds,
                )

                param.value = result.x
                q_value = -result.fun

                if not result.success:
                    print 'parameter values: ', param.value
                    print 'analytic derivative: ', self.evaluate_q_derivative(param.value, model, weights, param)
                    print 'numerical derivative: ', statsmodels.tools.numdiff.approx_fprime_cs(param.value, self.evaluate_q, args=(model, weights, param))
                    print result
                    raise OptimizeError(repr(result))
            
            print 'result:', param.value

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
            log_likelihood, weights = self.expectation_step(model)

            print 'log likelihood:', log_likelihood
            if log_likelihood_prev is not None:
                print 'log likelihood diff:', log_likelihood - log_likelihood_prev

            if log_likelihood_prev is not None and (log_likelihood_prev - log_likelihood) > self.likelihood_error_tol:
                self.error_message = 'log likelihood decreased from {} to {} for e step'.format(log_likelihood_prev, log_likelihood)
                return log_likelihood

            # Maximize Log likelihood with respect to haploid read depth
            try:
                q_value = self.maximization_step(model, weights, params)
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
