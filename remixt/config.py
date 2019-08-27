import remixt.defaults
import remixt.utils


def get_full_config(config):
    full_config = dict(vars(remixt.defaults))
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
    genome_fai = remixt.config.get_filename(config, ref_data_dir, 'genome_fai')
    chromosome_lengths = remixt.utils.read_chromosome_lengths(genome_fai)
    
    chromosomes = set(remixt.config.get_param(config, 'chromosomes'))

    filtered_chromosome_lengths = {}
    
    for chrom in chromosome_lengths.keys():
        if chrom in chromosomes:
            filtered_chromosome_lengths[chrom] = chromosome_lengths[chrom]
    
    return filtered_chromosome_lengths


def get_chromosomes(config, ref_data_dir):
    return list(get_chromosome_lengths(config, ref_data_dir).keys())
    
    
def get_sample_config(config, sample_id):
    sample_config = config.copy()
    sample_config.update(config.get('sample_specific', dict()).get(sample_id, dict()))
    return sample_config

