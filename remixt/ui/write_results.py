import argparse
import yaml
import pandas as pd


def write_results_tables(results_filename, cn_filename, brk_cn_filename, meta_filename, solution=None):
    """ Write the tables of a solution as TSV, and its statistics and parameters as yaml.

    Args:
        results_filename (str): results HDF5 from remixt collate
        cn_filename (str): output segment copy number table
        brk_cn_filename (str): output breakpoint copy number table
        meta_filename (str): output yaml of solution statistics and fit parameters

    KwArgs:
        solution (int): init id of the solution to write, default the optimal solution selected by collate
    """
    with pd.HDFStore(results_filename, 'r') as store:
        params = store['params']
        stats = store['stats'].set_index('init_id')

        if solution is None:
            solution = int(params['optimal_init_id'])
        elif solution not in stats.index:
            raise ValueError('no solution {}, available solutions are {}'.format(
                solution, ', '.join(str(a) for a in stats.index)))

        prefix = 'solutions/solution_{}/'.format(solution)
        cn = store[prefix + 'cn']
        brk_cn = store[prefix + 'brk_cn']
        h = store[prefix + 'h']
        mix = store[prefix + 'mix']

    cn.to_csv(cn_filename, sep='\t', index=False)
    brk_cn.to_csv(brk_cn_filename, sep='\t', index=False)

    def native(value):
        return value.item() if hasattr(value, 'item') else value

    metadata = {key: native(value) for key, value in stats.loc[solution].items()}
    metadata['init_id'] = int(solution)
    metadata['is_optimal'] = bool(solution == int(params['optimal_init_id']))
    metadata['params'] = {key: native(value) for key, value in params.items()}
    metadata['h'] = h.tolist()
    metadata['mix'] = mix.tolist()

    with open(meta_filename, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)


def add_arguments(argparser):
    argparser.add_argument('results_filename',
        help='ReMixT results filename')

    argparser.add_argument('cn_filename',
        help='Output segment copy number table filename')

    argparser.add_argument('brk_cn_filename',
        help='Output breakpoint copy number table filename')

    argparser.add_argument('meta_filename',
        help='Output meta data filename')

    argparser.add_argument('--solution', type=int, default=None,
        help='Init id of the solution to write, default the optimal solution selected by collate')

    argparser.set_defaults(func=write_results_tables)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    add_arguments(argparser)

    args = vars(argparser.parse_args())
    func = args.pop('func')
    func(**args)
