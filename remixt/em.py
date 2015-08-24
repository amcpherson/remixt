import scipy.optimize
import numpy as np



class OptimizeException(Exception):
    pass


class ExpectationMaximizationEstimator(object):

    def __init__(self, num_em_iter=100, posterior_tol=1e-3):
        """Create an expectation maximization estimator

        KwArgs:
            num_em_iter (int): number of em iterations
            posterior_tol (float): posterior increase tolerance

        """

        self.num_em_iter = num_em_iter
        self.posterior_tol = posterior_tol

        self.em_iter = None


    def evaluate_q(self, value, model, param, states, weights):
        """ Evaluate q function, expected value of complete data log likelihood
        with respect to conditional, given specific parameter value
        
        Args:
            value (numpy.array): parameter value
            model (object): probabilistic model to optimize
            param (str): name of parameter of model to optimize
            states (numpy.array): variable state matrix
            weights (numpy.array): state weights matrix

        Returns:
            numpy.array: expected value of complete data log likelihood

        """

        model.set_parameter(param, value)

        q_value = 0.0
        
        for s, w in zip(states, weights):
            
            log_likelihood = w * model.log_likelihood(s)
            log_likelihood[w == 0] = 0

            for n in zip(*np.where(np.isnan(log_likelihood))):
                raise ValueError('ll is nan', value=value, state=s[n], resp=w[n])

            q_value += np.sum(log_likelihood)

        return -q_value


    def evaluate_q_derivative(self, value, model, param, states, weights):
        """ Evaluate derivative of q function, expected complete data
        with respect to conditional, given specific parameter value
        
        Args:
            value (numpy.array): parameter value
            model (object): probabilistic model to optimize
            param (str): name of parameter of model to optimize
            states (numpy.array): variable state matrix
            weights (numpy.array): state weights matrix

        Returns:
            numpy.array: partial derivative of expected value of complete data log likelihood

        """

        model.set_parameter(param, value)
        
        q_derivative = np.zeros(value.shape)

        for s, w in zip(states, weights):

            log_likelihood_partial = (w.T * model.log_likelihood_partial(param, s).T).T
            
            for n in zip(*np.where(np.isnan(log_likelihood_partial))):
                raise ValueError('ll partial is nan', value=value, state=s[n], resp=w[n])

            if model.get_parameter_is_global(param):
                q_derivative += np.sum(log_likelihood_partial.T, axis=-1)
            else:
                q_derivative += np.sum(log_likelihood_partial, axis=-1)

        return -q_derivative


    def expectation_step(self, model):
        """ Expectation Step: Calculate weights for variable states
        
        Args:
            model (object): probabilistic model
        
        Returns:
            numpy.array: log posterior
            numpy.array: variable state matrix
            numpy.array: state weights matrix

        States matrix has shape (S,N,...) for N variables with S states
        Weights matrix has shape (S,N) for N variables with S states

        Weights are interpreted as posterior marginal probabilities.

        """

        log_posterior, states, weights = model.posterior_marginals()

        return log_posterior, states, weights


    def maximization_step(self, value_init, model, param, states, weights):
        """ Maximization Step.  Maximize Q with respect to a parameter.

        Args:
            value_init (numpy.array): initial parameter value
            model (object): probabilistic model to optimize
            param (str): name of parameter of model to optimize
            states (numpy.array): variable state matrix
            weights (numpy.array): state weights matrix

        Returns:
            numpy.array: optimal parameter maximizing Q

        """

        bounds = [model.get_parameter_bounds(param)] * value_init.shape[0]

        result = scipy.optimize.minimize(
            self.evaluate_q,
            value_init,
            method='L-BFGS-B',
            jac=self.evaluate_q_derivative,
            args=(model, param, states, weights),
            bounds=bounds,
            options={'ftol':1e-3})

        if not result.success:
            raise OptimizeException(result.message)

        return result.x


    def learn_param(self, model, param, value_init):
        """ Optimize h given an initial estimate.

        Args:
            model (object): probabilistic model to optimize
            param (str): name of parameter of model to optimize
            value_init (numpy.array): initial parameter value

        Returns:
            numpy.array: parameter value
            float: log posterior
            bool: converged

        The model object requires the following methods:
            def set_parameter(param, value)
            def get_parameter_bounds(param)
            def get_parameter_is_global(param)
            def log_likelihood(state)
            def log_likelihood_partial(param, state)

        """

        value = value_init

        model.set_parameter(param, value)

        log_posterior_prev = None

        converged = False

        for self.em_iter in xrange(self.num_em_iter):

            # Maximize Log likelihood with respect to copy number
            log_posterior, states, weights = self.expectation_step(model)

            # Maximize Log likelihood with respect to haploid read depth
            value = self.maximization_step(value, model, param, states, weights)

            print value, log_posterior

            model.set_parameter(param, value)

            if log_posterior_prev is not None and abs(log_posterior_prev - log_posterior) < self.posterior_tol:
                converged = True
                break

            log_posterior_prev = log_posterior

        return value, log_posterior, converged



class HardAssignmentEstimator(ExpectationMaximizationEstimator):

    def __init__(self, num_em_iter=100, posterior_tol=1e-3):
        """ Hard assignment EM

        KwArgs:
            num_em_iter (int): number of em iterations
            posterior_tol (float): posterior increase tolerance

        """

        ExpectationMaximizationEstimator.__init__(self, num_em_iter=num_em_iter, posterior_tol=posterior_tol)

        self.state = None


    def expectation_step(self, model):
        """ Override expectation step for hard assignment.
        
        Args:
            model (object): probabilistic model
        
        Returns:
            numpy.array: log posterior
            numpy.array: variable state matrix
            numpy.array: state weights matrix

        States matrix has shape (1,N,...) for N variables
        Weights matrix has shape (1,N) for N variables

        """

        log_posterior, self.state = self.assignment_step(model)

        S = 1
        N = self.state.shape[0]

        states = self.state[np.newaxis]
        weights = np.ones((S, N))

        return log_posterior, states, weights


    def assignment_step(self, model):
        """ Assignment Step: Calculate optimal state assignment of variables
        
        Args:
            model (object): probabilistic model
        
        Returns:
            numpy.array: log posterior
            numpy.array: variable state

        State array has shape (N,...) for N variables

        """

        log_posterior, state = model.optimal_state()

        return log_posterior, state


