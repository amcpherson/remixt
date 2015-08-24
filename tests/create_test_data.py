import os
import pickle
import numpy as np

import remixt.simulations.experiment


np.random.seed(2014)


def generate_test_experiment(experiment_filename):

    params = {}

    rh_sampler = remixt.simulations.experiment.RearrangementHistorySampler(params)
    gc_sampler = remixt.simulations.experiment.GenomeCollectionSampler(rh_sampler, params)

    gc = gc_sampler.sample_genome_collection()

    gm_sampler = remixt.simulations.experiment.GenomeMixtureSampler(params)

    gm = gm_sampler.sample_genome_mixture(gc)

    experiment_sampler = remixt.simulations.experiment.ExperimentSampler(params)

    experiment = experiment_sampler.sample_experiment(gm)

    with open(experiment_filename, 'w') as experiment_file:
        pickle.dump(experiment, experiment_file)


if __name__ == '__main__':

    script_directory = os.path.realpath(os.path.dirname(__file__))
    experiment_filename = os.path.join(script_directory, 'test_experiment.pickle')

    generate_test_experiment(experiment_filename)


