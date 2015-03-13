import demix.wrappers.wrapdemix
import demix.wrappers.wraptitan
import demix.wrappers.wraptheta

catalog = dict()
catalog['demix'] = demix.wrappers.wrapdemix.DemixTool
catalog['titan'] = demix.wrappers.wraptitan.TitanTool
catalog['theta'] = demix.wrappers.wraptheta.ThetaTool

