import remixt.defaults
import remixt.utils


def get_full_config(config):
    full_config = dict(vars(remixt.defaults))
    genome_version = config.get('ensembl_genome_version', remixt.defaults.ensembl_genome_version)
    full_config.update(remixt.defaults.genome_versions.get(genome_version, {}))
    full_config.update(config)
    return full_config


def get_param(config, name):
    return get_full_config(config)[name]


def get_filename(config, ref_data_dir, name, **kwargs):
    full_config = get_full_config(config)
    full_config.update(kwargs)
    full_config['ref_data_dir'] = ref_data_dir
    if name+'_filename' in full_config:
        return full_config[name+'_filename']
    elif name+'_template' in full_config:
        return full_config[name+'_template'].format(**full_config)


def get_chromosome_lengths(config, ref_data_dir):
    return remixt.utils.read_selected_chromosome_lengths(
        get_filename(config, ref_data_dir, 'genome_fai'),
        get_param(config, 'chromosomes'),
        chr_name_prefix=get_param(config, 'chr_name_prefix'),
    )


def get_chromosomes(config, ref_data_dir):
    return list(get_chromosome_lengths(config, ref_data_dir).keys())
    
    
def get_sample_config(config, sample_id):
    sample_config = config.copy()
    sample_config.update(config.get('sample_specific', dict()).get(sample_id, dict()))
    return sample_config

