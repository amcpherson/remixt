import numpy as np
import statsmodels.tools.numdiff


def assert_grad_correct(func, grad, x0, *args, **kwargs):
    """ Assert correct gradiant compared to finite difference approximation
    """

    analytic_fprime = grad(x0, *args)
    approx_fprime = statsmodels.tools.numdiff.approx_fprime_cs(x0, func, args=args)

    np.testing.assert_almost_equal(analytic_fprime, approx_fprime, 5)
