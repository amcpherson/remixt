import os
import pickle
import numpy as np

import remixt.simulations.experiment
import remixt.cn_plot


np.random.seed(2016)


def generate_experiment(experiment_filename, experiment_plot_filename):

    params = {
        'N': 5000,
        'frac_clone': [0.4, 0.2],
        'emission_model': 'normal',
        'num_ancestral_events': 100,
        'num_descendent_events': 50,
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


if __name__ == '__main__':

    script_directory = os.path.realpath(os.path.dirname(__file__))
    experiment_filename = os.path.join(script_directory, 'test_experiment.pickle')
    experiment_plot_filename = os.path.join(script_directory, 'test_experiment_plot.pdf')

    generate_experiment(experiment_filename, experiment_plot_filename)
