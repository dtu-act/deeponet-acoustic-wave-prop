# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import os
import matplotlib.pyplot as plt
from pathlib import Path
import collections
from setup.configurations import setupPlotParams
import numpy as np

from setup.settings import SimulationSettings
import setup.parsers as parsers

# https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html

setupPlotParams()

LossLogger = collections.namedtuple('LossLogger', ['loss_train', 'loss_val', 'nIter'])

output_dir = "/work3/nibor/data/deeponet/output_2D_FA"
figs_dir = output_dir

sim_tag = "loss_spectral_bias_best_mlp"
ids = [("1_baseline_tanh_mlp_pos", "tanh MLP Positional"),
       ("1_baseline_relu_mlp_gaussian", "relu MLP Gaussian"),
       ("1_baseline_sine_mlp", "sine MLP")]

# ids = [("cube_6ppw", "Cubic"),
#        ("Lshape_6ppw", "L-shape"),
#        ("furnished_6ppw", "Furnished"),
#        ("bilbao_6ppw", "Dome"),
#        ("bilbao_6ppw_1stquad", "Dome 1/4")]


def writeLoss(simulation_str, err_train, err_val, f):
    f.write('------')
    f.write(simulation_str)
    f.write('\nMSE train: {:.1E}\n'.format(err_train))
    f.write('MSE val: {:.1E}\n'.format(err_val))

def loadLosses(log_dir):    
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    nIter = list(map(lambda e: e.step, event_acc.Scalars('Loss/train/loss')))
    loss_train = list(map(lambda e: e.value, event_acc.Scalars('Loss/train/loss')))
    loss_val = list(map(lambda e: e.value, event_acc.Scalars('Loss/val/loss')))

    return LossLogger(loss_train,loss_val,nIter)

def plotConvergenceMultipleModels(ids, figs_dir, skip=1):
    path_conv_dir = os.path.join(figs_dir,"convergence_plots")
    Path(path_conv_dir).mkdir(parents=False, exist_ok=True)
    
    #plt.figure(figsize = (6,4))
    fig, ax = plt.subplots(figsize = (6,4), dpi=400)

    #colors = ["blue", "orange", "green", "red", "red"]
    colors = ["b", "g", "r", "c", "m", "y"]
    
    error_filepath = os.path.join(path_conv_dir, sim_tag + ".txt")

    with open(error_filepath, 'w') as f:
        for i, id in enumerate(ids):
            settings_path = os.path.join(output_dir, f"{id[0]}/settings.json")

            settings_dict = parsers.parseSettings(settings_path)
            settings = SimulationSettings(settings_dict, output_dir=output_dir)
            loss_logger = loadLosses(settings.dirs.models_dir)

            color_i = colors[i % len(colors)]
            
            writeLoss(id[1], loss_logger.loss_train[-1], loss_logger.loss_val[-1], f)

            # i = np.where(np.array(loss_logger.nIter) > 70000)[0]
            # i = len(loss_logger.nIter) if len(i) == 0 else i[0]
            i = len(loss_logger.nIter)

            plt.plot(loss_logger.nIter[:i:skip], loss_logger.loss_train[:i:skip], label=id[1])
            #plt.plot(loss_logger.nIter[:i:skip], loss_logger.loss_train[:i:skip], label=f"{id[1]} train", color=color_i)
            # plt.plot(loss_logger.nIter[:i:skip], loss_logger.loss_val[:i:skip], label=f"{id[1]} val", linestyle="dashed", linewidth=2, color=color_i)
            
    
    # ax.xaxis.set_ticks([0,30000,50000,60000,70000])
    # ax.set_xticklabels(['0','30k','50k','60k','70k'])
    ax.xaxis.set_ticks([0,40000,80000])
    ax.set_xticklabels(['0','40k','80k'])
    ax.set_ylim([1e-4,9e-3])
    ax.set_xlabel('Iterations')
    ax.set_ylabel(r'$L_2$ loss')    
    ax.set_yscale('log')
    ax.legend(loc='center left',  fancybox=True, bbox_to_anchor=(1, 0.5)) #, ncol=5
    # ax.legend()
    ax.grid()
    # fig.tight_layout()

    if figs_dir == None:
        fig.show()
    else:        
        fig_path = os.path.join(path_conv_dir, sim_tag + '.png')
        fig.savefig(fig_path,bbox_inches='tight',pad_inches=0)

plotConvergenceMultipleModels(ids, figs_dir, skip=1)