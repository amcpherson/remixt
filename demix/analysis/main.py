import pickle
import numpy as np
import pandas as pd


def optimize_h(model_filename, experiment_filename, h_init):

    with open(experiment_filename, 'r') as experiment_file:
        exp = pickle.load(experiment_file)

    with open(model_filename, 'r') as model_file:
        model = pickle.load(model_file)

    h, log_posterior, converged = model.optimize_h(exp.x, exp.l, np.array(h_init))

    return list(h), log_posterior, converged


def tabulate_h(h_table_filename, h_opt):

    h_table = list()

    for idx, (h, log_posterior, converged) in h_opt.iteritems():

        h_row = dict()

        h_row['idx'] = idx
        h_row['log_posterior'] = log_posterior
        h_row['converged'] = converged

        for h_idx, h_value in enumerate(h):
            h_row['h_{}'.format(h_idx)] = h_value

        h_table.append(h_row)

    h_table = pd.DataFrame(h_table)

    h_table.to_csv(h_table_filename, sep='\t', index=False)


def infer_cn(cn_table_filename, brk_cn_table_filename, experiment_plot_filename,
        mix_filename, model_filename, experiment_filename, h_table_filename):

    with open(experiment_filename, 'r') as experiment_file:
        exp = pickle.load(experiment_file)

    with open(model_filename, 'r') as model_file:
        model = pickle.load(model_file)

    h_table = pd.read_csv(h_table_filename, sep='\t')

    h = h_table.sort('log_posterior', ascending=False).iloc[0][['h_{0}'.format(idx) for idx in xrange(3)]].values.astype(float)

    with open(mix_filename, 'w') as mix_file:
        mix_file.write('\t'.join(list(h / h.sum())))

    model.e_step_method = 'genomegraph'

    cn, brk_cn = model.decode(exp.x, exp.l, h)

    fig = cn_plot.experiment_plot(exp, cn, h, model.p)

    fig.savefig(experiment_plot_filename, format='pdf', bbox_inches='tight')

    cn_table = pd.DataFrame({
        'chromosome':exp.segment_chromosome_id,
        'start':exp.segment_start,
        'end':exp.segment_end,
    })

    for m in xrange(1, model.M):
        for l in xrange(2):
            cn_table['cn_{0}_{1}'.format(m, l)] = cn[:,m,l]

    cn_table.to_csv(cn_table_filename, sep='\t', index=False, header=True)

    brk_cn_table = list()

    for bp, bp_cn in brk_cn.iteritems():
        ((n_1, ell_1), side_1), ((n_2, ell_2), side_2) = bp
        bp_id = exp.breakpoint_ids[frozenset([(n_1, side_1), (n_2, side_2)])]
        brk_cn_table.append([bp_id, ell_1, ell_2] + list(bp_cn[1:]))

    brk_cn_cols = ['prediction_id', 'allele_1', 'allele_2']
    for m in xrange(1, model.M):
        brk_cn_cols.append('cn_{0}'.format(m))
    brk_cn_table = pd.DataFrame(brk_cn_table, columns=brk_cn_cols)

    brk_cn_table.to_csv(brk_cn_table_filename, sep='\t', index=False, header=True)



