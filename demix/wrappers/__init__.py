import demix.wrappers.demix
import demix.wrappers.titan
import demix.wrappers.theta

catalog = dict()
catalog['demix'] = demix.wrappers.demix.DemixWrapper
catalog['titan'] = demix.wrappers.titan.TitanWrapper
catalog['theta'] = demix.wrappers.theta.ThetaWrapper

