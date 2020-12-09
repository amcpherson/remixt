import argparse
import yaml
import pandas as pd
import numpy as np



def write_results_tables(**args):
    store = pd.HDFStore(args['results_filename'], 'r')

    stats = store['stats']

    # Filter high proportion subclonal
    stats = stats[stats['proportion_divergent'] <= args['max_proportion_divergent']]

    # Filter based on ploidy range
    if args.get('max_ploidy') is not None:
        stats = stats[stats['ploidy'] < args['max_ploidy']]
    if args.get('min_ploidy') is not None:
        stats = stats[stats['ploidy'] > args['min_ploidy']]

    if stats.empty:
        raise ValueError('filters to restrictive, no solutions')

    # Select highest elbo solution
    stats = stats.sort_values('elbo', ascending=False).iloc[0]
    solution = stats['init_id']

    cn = store['solutions/solution_{0}/cn'.format(solution)]
    brk_cn = store['solutions/solution_{0}/brk_cn'.format(solution)]
    h = store['solutions/solution_{0}/h'.format(solution)]
    mix = store['solutions/solution_{0}/mix'.format(solution)]

    cn.to_csv(args['cn_filename'], sep='\t', index=False)
    brk_cn.to_csv(args['brk_cn_filename'], sep='\t', index=False)

    metadata = dict()
    for key, value in stats.items():
        try:
            metadata[key] = np.asscalar(value)
        except AttributeError:
            metadata[key] = value
    metadata['h'] = list(h.tolist())
    metadata['mix'] = list(mix.tolist())

    with open(args['meta_filename'], 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    store.close()


def add_arguments(argparser):
    argparser.add_argument('results_filename',
        help='ReMixT results filename')

    argparser.add_argument('cn_filename',
        help='Output segment copy number table filename')

    argparser.add_argument('brk_cn_filename',
        help='Output breakpoint copy number table filename')

    argparser.add_argument('meta_filename',
        help='Output meta data filename')

    argparser.add_argument('--max_ploidy', type=float, default=None,
        help='Maximum ploidy')

    argparser.add_argument('--min_ploidy', type=float, default=None,
        help='Minimum ploidy')

    argparser.add_argument('--max_proportion_divergent', type=float, default=0.5,
        help='Maximum proportion of the genome divergent')

    argparser.set_defaults(func=write_results_tables)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    add_arguments(argparser)

    args = vars(argparser.parse_args())
    func = args.pop('func')
    func(**args)


