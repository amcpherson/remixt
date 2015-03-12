import collections
import itertools
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import demix.cn_model




def create_model(model_filename, experiment_filename):

    with open(experiment_filename, 'r') as experiment_file:
        exp = pickle.load(experiment_file)

    model = demix.cn_model.CopyNumberModel(3, exp.adjacencies, exp.breakpoints)
    model.emission_model = 'negbin'
    model.e_step_method = 'forwardbackward'
    model.total_cn = True

    model.infer_offline_parameters(exp.x, exp.l)

    with open(model_filename, 'w') as model_file:
        pickle.dump(model, model_file)


def create_candidate_h(model_filename, experiment_filename, candidate_h_plot_filename):

    with open(model_filename, 'r') as model_file:
        model = pickle.load(model_file)

    with open(experiment_filename, 'r') as experiment_file:
        exp = pickle.load(experiment_file)

    fig = plt.figure(figsize=(8,8))

    ax = plt.subplot(1, 1, 1)

    candidate_h_init = model.candidate_h(exp.x, exp.l, ax=ax)

    fig.savefig(candidate_h_plot_filename, format='pdf')

    return dict(enumerate([tuple(h) for h in candidate_h_init]))


