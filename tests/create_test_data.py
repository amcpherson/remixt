import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import remixt.simulations.experiment
import remixt.cn_plot
import remixt.analysis.experiment


np.random.seed(2019)


def generate_experiment(experiment_filename, experiment_plot_filename, scatter_plot_filename):

    params = {
        'N': 5000,
        'frac_normal': 0.4,
        'frac_clone_1': 0.2,
        'num_ancestral_events': 50,
        'num_descendent_events': 25,
        'num_false_breakpoints': 0,
        'ploidy_gaussian_mean': 2.5,
    }

    rh_sampler = remixt.simulations.experiment.RearrangementHistorySampler(params)
    gc_sampler = remixt.simulations.experiment.GenomeCollectionSampler(rh_sampler, params)

    gc = gc_sampler.sample_genome_collection()

    gm_sampler = remixt.simulations.experiment.GenomeMixtureSampler(params)

    gm = gm_sampler.sample_genome_mixture(gc)

    experiment_sampler = remixt.simulations.experiment.ExperimentSampler(params)

    experiment = experiment_sampler.sample_experiment(gm)

    with open(experiment_filename, 'w') as experiment_file:
        pickle.dump(experiment, experiment_file)

    fig = remixt.cn_plot.experiment_plot(experiment, experiment.cn, experiment.h, maxcopies=6)
    fig.savefig(experiment_plot_filename)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    cnv = remixt.analysis.experiment.create_cn_table(experiment, experiment.cn, experiment.h)
    remixt.cn_plot.plot_cnv_scatter(ax, cnv, major_col='major_raw', minor_col='minor_raw')
    fig.savefig(scatter_plot_filename)


if __name__ == '__main__':

    script_directory = os.path.realpath(os.path.dirname(__file__))
    experiment_filename = os.path.join(script_directory, 'test_experiment.pickle')
    experiment_plot_filename = os.path.join(script_directory, 'test_experiment_plot.pdf')
    scatter_plot_filename = os.path.join(script_directory, 'test_scatter_plot.pdf')

    generate_experiment(experiment_filename, experiment_plot_filename, scatter_plot_filename)
