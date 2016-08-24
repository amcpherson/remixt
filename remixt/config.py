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
    
    for chrom in chromosome_lengths.keys():
        if chrom not in chromosomes:
            del chromosome_lengths[chrom]
    
    return chromosome_lengths


def get_chromosomes(config, ref_data_dir):
    return get_chromosome_lengths(config, ref_data_dir).keys()

