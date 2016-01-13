import remixt.defaults


def get_full_config(config):
    full_config = dict(vars(remixt.defaults))
    full_config.update(config)
    return full_config


def get_param(config, name):
    return get_full_config(config)[name]


def get_filename(config, name, **kwargs):
    full_config = get_full_config(config)
    full_config.update(kwargs)
    if name+'_filename' in full_config:
        return full_config[name+'_filename']
    elif name+'_template' in full_config:
        return full_config[name+'_template'].format(**full_config)

