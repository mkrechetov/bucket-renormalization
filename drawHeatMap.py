import argparse
import os
import random

from matplotlib.gridspec import GridSpec

import protocols
import utils
import networkx as nx
import matplotlib.pyplot as plt
from gmi import GMI
import pandas as pd
from graphical_model import *
from factor import *
from generate_model import generate_complete_gmi, generate_complete
from bucket_elimination import BucketElimination
from bucket_renormalization import BucketRenormalization
import itertools
import traceback
import numpy as np

from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.patches import Ellipse
import numpy.random as rnd
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def drawProbabilityHeatmap(passedFileName,tractUVCoords,rawSeattleImage,probabilities1,probabilities2,probabilities3,probabilities4,probabilities5):
    fig, mainAxe = plt.subplots(figsize=(19.20, 4.5),constrained_layout=True)
    mainAxe.set_visible(False)
    fig.suptitle('Marginal Probabilities initial infection=[0]', fontsize=16)
    gs = GridSpec(1, 5, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    ax1.set_title('BETA=3.0_MU=80_EPS=0.4')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    ax2.set_title('BETA=3.0_MU=90_EPS=0.4')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axes.xaxis.set_visible(False)
    ax3.axes.yaxis.set_visible(False)
    ax3.set_title('BETA=3.0_MU=100_EPS=0.4')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axes.xaxis.set_visible(False)
    ax4.axes.yaxis.set_visible(False)
    ax4.set_title('BETA=3.0_MU=110_EPS=0.4')

    ax5 = fig.add_subplot(gs[0, 4])
    ax5.axes.xaxis.set_visible(False)
    ax5.axes.yaxis.set_visible(False)
    ax5.set_title('BETA=3.0_MU=120_EPS=0.4')

    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap(
        [c('blue'), c('red')])
    norm = mpl.colors.Normalize(vmin=probabilities1.T.min(), vmax=probabilities1.T.max())

    divider = make_axes_locatable(mainAxe)
    cax = divider.append_axes('right', size='1%', pad=0.1)

    mpl.colorbar.ColorbarBase(cax, cmap=rvb,
                              norm=norm,
                              orientation='vertical')

    ax1.imshow(rawSeattleImage,cmap=rvb)
    for index in range(probabilities1.shape[1]):
        ax1.add_patch(Ellipse((tractUVCoords.iloc[index][1],tractUVCoords.iloc[index][2]), width=11, height=11,
                             edgecolor='None',
                             facecolor=(probabilities1.iloc[0][index],0,1-probabilities1.iloc[0][index],1),
                             linewidth=1))
    ax2.imshow(rawSeattleImage, cmap=rvb)
    for index in range(probabilities2.shape[1]):
        ax2.add_patch(Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=11, height=11,
                              edgecolor='None',
                              facecolor=(probabilities2.iloc[0][index], 0, 1 - probabilities2.iloc[0][index], 1),
                              linewidth=1))
    ax3.imshow(rawSeattleImage, cmap=rvb)
    for index in range(probabilities3.shape[1]):
        ax3.add_patch(Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=11, height=11,
                              edgecolor='None',
                              facecolor=(probabilities3.iloc[0][index], 0, 1 - probabilities3.iloc[0][index], 1),
                              linewidth=1))
    ax4.imshow(rawSeattleImage, cmap=rvb)
    for index in range(probabilities4.shape[1]):
        ax4.add_patch(Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=11, height=11,
                              edgecolor='None',
                              facecolor=(probabilities4.iloc[0][index], 0, 1 - probabilities4.iloc[0][index], 1),
                              linewidth=1))
    ax5.imshow(rawSeattleImage, cmap=rvb)
    for index in range(probabilities5.shape[1]):
        ax5.add_patch(Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=11, height=11,
                              edgecolor='None',
                              facecolor=(probabilities5.iloc[0][index], 0, 1 - probabilities5.iloc[0][index], 1),
                              linewidth=1))

    plt.tight_layout()
    plt.show()
    fig.savefig("./results_experiment2/"+passedFileName+".png")#AUTOMATICALLY SAVES THE IMAGE FILES IN RESULTS FOLDER

def renormalizeProbability(input):
    output=pd.DataFrame()
    max_value=0
    min_value=1000000
    for i in range(input.shape[0]):  # TEST RANDOM PROBABILITIES
        if input.iloc[i][1]<min_value:
            min_value=input.iloc[i][1]
        if input.iloc[i][1]>max_value:
            max_value=input.iloc[i][1]
    for i in range(input.shape[0]):  # TEST RANDOM PROBABILITIES
        output[str(i)] = np.array([1-(input.iloc[i][1]-min_value)/(max_value-min_value)])
    return output


#\/\/\/ TESTING FOR SAVE PROBABILITY HEATMAP TO FILE
#TO DO: MIX THIS TEST CODE WITH HPC RELATED CODE.
tractUVCoords = pd.read_csv('./seattle/tractUVCoordinates.csv')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES
rawSeattleImage=mpimg.imread('./seattle/SeattleRawImage.jpg')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES

#TEST RANDOM PROBABILITIES. PROBABILITIES SHOULD BE IN A SINGLE ROW
# testProbabilities = pd.DataFrame()#TEST RANDOM PROBABILITIES
# for i in range(tractUVCoords.shape[0]):#TEST RANDOM PROBABILITIES
#     testProbabilities[str(i)] = rnd.rand(1)#TEST RANDOM PROBABILITIES
probabilities1 = pd.read_csv('./results_experiment2/seattle_marginal_probabilities_init_inf=[0]_BETA=3.0_MU=80_EPS=0.4.csv')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES
probabilities1 = renormalizeProbability(probabilities1)

probabilities2 = pd.read_csv('./results_experiment2/seattle_marginal_probabilities_init_inf=[0]_BETA=3.0_MU=90_EPS=0.4.csv')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES
probabilities2 = renormalizeProbability(probabilities2)

probabilities3 = pd.read_csv('./results_experiment2/seattle_marginal_probabilities_init_inf=[0]_BETA=3.0_MU=100_EPS=0.4.csv')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES
probabilities3 = renormalizeProbability(probabilities3)

probabilities4 = pd.read_csv('./results_experiment2/seattle_marginal_probabilities_init_inf=[0]_BETA=3.0_MU=110_EPS=0.4.csv')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES
probabilities4 = renormalizeProbability(probabilities4)

probabilities5 = pd.read_csv('./results_experiment2/seattle_marginal_probabilities_init_inf=[0]_BETA=3.0_MU=120_EPS=0.4.csv')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES
probabilities5 = renormalizeProbability(probabilities5)

#A NAMING SCHEMA IS REQUIRED TO REPLACE "test". THE DIRECTORY IS SET INSIDE THE FUNCTION.
#tractUVCoords, AND rawSeattleImage SHOULD BE READ ONCE AND USED MULTIPLE TIMES.
test_name = "beta_3_mu_80_to_120_EPS_0_4"
drawProbabilityHeatmap(test_name,tractUVCoords,rawSeattleImage,probabilities1,probabilities2,probabilities3,probabilities4,probabilities5)
#^^^ TESTING FOR SAVE PROBABILITY HEATMAP TO FILE