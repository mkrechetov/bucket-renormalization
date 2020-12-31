from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.patches import Ellipse
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

def drawProbabilityHeatmap(passedFileName,tractUVCoords,rawSeattleImage,betas,mu,eps,probabilities):
    numPlots=0
    for b in betas:
        for m in mu:
            for e in eps:
                numPlots=numPlots+1

    if numPlots<=5:
        fig, mainAxe = plt.subplots(figsize=(19.20, 4.6), constrained_layout=True)
        mainAxe.set_visible(False)
        fig.suptitle('Marginal Probabilities initial infection=[0]', fontsize=16)
        gs = GridSpec(1, numPlots, figure=fig)

        c = mcolors.ColorConverter().to_rgb
        rvb = make_colormap(
            [c('blue'), c('red')])


        counter = 0
        #axes=[]
        for b in betas:
            for m in mu:
                for e in eps:
                    ax = fig.add_subplot(gs[0, counter])
                    #ax.axes.xaxis.set_visible(False)
                    #ax.axes.yaxis.set_visible(False)
                    ax.set_title('BETA=' + b + '_MU=' + m + '_EPS='+e)
                    #axes.append(ax)

                    ax.imshow(rawSeattleImage, interpolation="nearest")

                    # inset axes....
                    axins = ax.inset_axes([0.01, 0.5, 0.47, 0.47])
                    axins.imshow(rawSeattleImage, interpolation="nearest",
                                 origin="lower")
                    # sub region of the original image
                    x1, x2, y1, y2 = 450, 650, 400, 700
                    axins.set_xlim(x1, x2)
                    axins.set_ylim(y2, y1)
                    axins.axes.xaxis.set_visible(False)
                    axins.axes.yaxis.set_visible(False)
                    axins.set_xticklabels('')
                    axins.set_yticklabels('')

                    ax.indicate_inset_zoom(axins)


                    #ax.imshow(rawSeattleImage, cmap=rvb)

                    for index in range(probabilities[counter].shape[1]):
                        ax.add_patch(
                            Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=11, height=11,
                                    edgecolor='None',
                                    facecolor=(probabilities[counter].iloc[0][index], 0, 1 - probabilities[counter].iloc[0][index], 1),
                                    linewidth=1))
                    for index in range(probabilities[counter].shape[1]):
                        axins.add_patch(
                            Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=15, height=15,
                                    edgecolor='None',
                                    facecolor=(probabilities[counter].iloc[0][index], 0, 1 - probabilities[counter].iloc[0][index], 1),
                                    linewidth=1))

                    counter = counter+1

    elif numPlots==9:
        fig, mainAxe = plt.subplots(figsize=(19.20, 10), constrained_layout=True)
        mainAxe.set_visible(False)
        fig.suptitle('Marginal Probabilities initial infection=[0]', fontsize=16)
        gs = GridSpec(3, 3, figure=fig)

        c = mcolors.ColorConverter().to_rgb
        rvb = make_colormap(
            [c('blue'), c('red')])

        counterP=0
        counterR = 0
        counterC=0
        # axes=[]
        for b in betas:
            for m in mu:
                for e in eps:
                    ax = fig.add_subplot(gs[counterR, counterC])
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                    ax.set_title('BETA=' + b + '_MU=' + m + '_EPS=' + e)
                    # axes.append(ax)

                    ax.imshow(rawSeattleImage, interpolation="nearest")

                    # inset axes....
                    axins = ax.inset_axes([0.01, 0.5, 0.47, 0.47])
                    axins.imshow(rawSeattleImage, interpolation="nearest",
                                 origin="lower")
                    # sub region of the original image
                    x1, x2, y1, y2 = 450, 650, 400, 700
                    axins.set_xlim(x1, x2)
                    axins.set_ylim(y2, y1)
                    axins.axes.xaxis.set_visible(False)
                    axins.axes.yaxis.set_visible(False)
                    axins.set_xticklabels('')
                    axins.set_yticklabels('')

                    ax.indicate_inset_zoom(axins)

                    #ax.imshow(rawSeattleImage, cmap=rvb)
                    for index in range(probabilities[counterP].shape[1]):
                        ax.add_patch(
                            Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=11, height=11,
                                    edgecolor='None',
                                    facecolor=(
                                    probabilities[counterP].iloc[0][index], 0, 1 - probabilities[counterP].iloc[0][index],
                                    1),
                                    linewidth=1))
                    for index in range(probabilities[counterP].shape[1]):
                        axins.add_patch(
                            Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=15, height=15,
                                    edgecolor='None',
                                    facecolor=(probabilities[counterP].iloc[0][index], 0, 1 - probabilities[counterP].iloc[0][index], 1),
                                    linewidth=1))
                    counterR=counterR+1
                    if counterR>=3:
                        counterC=counterC+1
                        counterR=0

                    counterP = counterP + 1
    else:
        print('not implemented')

    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    divider = make_axes_locatable(mainAxe)
    cax = divider.append_axes('right', size='1%', pad='2%')

    mpl.colorbar.ColorbarBase(cax, cmap=rvb,
                              norm=norm,
                              orientation='vertical')

    plt.tight_layout()
    plt.show()
    fig.savefig(passedFileName+".png")#AUTOMATICALLY SAVES THE IMAGE FILES IN RESULTS FOLDER

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


#TO DO: MIX THIS TEST CODE WITH HPC RELATED CODE.
tractUVCoords = pd.read_csv('./seattle/tractUVCoordinates.csv')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES
rawSeattleImage=mpimg.imread('./seattle/SeattleRawImage1.png')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES

#\/\/\/ SAVE PROBABILITY HEATMAP TO FILE experiment 2
#TEST RANDOM PROBABILITIES. PROBABILITIES SHOULD BE IN A SINGLE ROW
# testProbabilities = pd.DataFrame()#TEST RANDOM PROBABILITIES
# for i in range(tractUVCoords.shape[0]):#TEST RANDOM PROBABILITIES
#     testProbabilities[str(i)] = rnd.rand(1)#TEST RANDOM PROBABILITIES
betas=['3.0']
mu=['80','90','100','110','120']
eps=['0.4']
probabilities=[]
for b in betas:
    for m in mu:
        for e in eps:
            file_name='./results_experiment2/seattle_marginal_probabilities_init_inf=[0]_BETA='+b+'_MU='+m+'_EPS='+e+'.csv'
            temp=pd.read_csv(file_name)
            temp = renormalizeProbability(temp)
            probabilities.append(temp)

#A NAMING SCHEMA IS REQUIRED TO REPLACE "test". THE DIRECTORY IS SET INSIDE THE FUNCTION.
#tractUVCoords, AND rawSeattleImage SHOULD BE READ ONCE AND USED MULTIPLE TIMES.
test_name = "./results_experiment2/beta_3_mu_80_to_120_EPS_0_4"
drawProbabilityHeatmap(test_name,tractUVCoords,rawSeattleImage,betas,mu,eps,probabilities)
#^^^ SAVE PROBABILITY HEATMAP TO FILE experiment 2

#\/\/\/ SAVE PROBABILITY HEATMAP TO FILE experiment 2
#TEST RANDOM PROBABILITIES. PROBABILITIES SHOULD BE IN A SINGLE ROW
# testProbabilities = pd.DataFrame()#TEST RANDOM PROBABILITIES
# for i in range(tractUVCoords.shape[0]):#TEST RANDOM PROBABILITIES
#     testProbabilities[str(i)] = rnd.rand(1)#TEST RANDOM PROBABILITIES
betas=['5.0']
mu=['80','90','100','110','120']
eps=['0.4']
probabilities=[]
for b in betas:
    for m in mu:
        for e in eps:
            file_name='./results_experiment2/seattle_marginal_probabilities_init_inf=[0]_BETA='+b+'_MU='+m+'_EPS='+e+'.csv'
            temp=pd.read_csv(file_name)
            temp = renormalizeProbability(temp)
            probabilities.append(temp)

#A NAMING SCHEMA IS REQUIRED TO REPLACE "test". THE DIRECTORY IS SET INSIDE THE FUNCTION.
#tractUVCoords, AND rawSeattleImage SHOULD BE READ ONCE AND USED MULTIPLE TIMES.
test_name = "./results_experiment2/beta_5_mu_80_to_120_EPS_0_4"
drawProbabilityHeatmap(test_name,tractUVCoords,rawSeattleImage,betas,mu,eps,probabilities)
#^^^ SAVE PROBABILITY HEATMAP TO FILE experiment 2

#\/\/\/ SAVE PROBABILITY HEATMAP TO FILE experiment 1
#TEST RANDOM PROBABILITIES. PROBABILITIES SHOULD BE IN A SINGLE ROW
betas=['3.0']
mu=['100','120','140']
eps=['0.3','0.4','0.5']
probabilities=[]
for b in betas:
    for m in mu:
        for e in eps:
            file_name='./results_experiment1/seattle_marginal_probabilities_init_inf=[0]_BETA='+b+'_MU='+m+'_EPS='+e+'.csv'
            temp=pd.read_csv(file_name)
            temp = renormalizeProbability(temp)
            probabilities.append(temp)

#A NAMING SCHEMA IS REQUIRED TO REPLACE "test". THE DIRECTORY IS SET INSIDE THE FUNCTION.
#tractUVCoords, AND rawSeattleImage SHOULD BE READ ONCE AND USED MULTIPLE TIMES.
test_name = "./results_experiment1/beta_3_mu_100_to_140_EPS_0_3_to_0_5"
drawProbabilityHeatmap(test_name,tractUVCoords,rawSeattleImage,betas,mu,eps,probabilities)
#^^^ SAVE PROBABILITY HEATMAP TO FILE experiment 1

#\/\/\/ SAVE PROBABILITY HEATMAP TO FILE experiment 1
#TEST RANDOM PROBABILITIES. PROBABILITIES SHOULD BE IN A SINGLE ROW
betas=['5.0']
mu=['100','120','140']
eps=['0.3','0.4','0.5']
probabilities=[]
for b in betas:
    for m in mu:
        for e in eps:
            file_name='./results_experiment1/seattle_marginal_probabilities_init_inf=[0]_BETA='+b+'_MU='+m+'_EPS='+e+'.csv'
            temp=pd.read_csv(file_name)
            temp = renormalizeProbability(temp)
            probabilities.append(temp)

#A NAMING SCHEMA IS REQUIRED TO REPLACE "test". THE DIRECTORY IS SET INSIDE THE FUNCTION.
#tractUVCoords, AND rawSeattleImage SHOULD BE READ ONCE AND USED MULTIPLE TIMES.
test_name = "./results_experiment1/beta_5_mu_100_to_140_EPS_0_3_to_0_5"
drawProbabilityHeatmap(test_name,tractUVCoords,rawSeattleImage,betas,mu,eps,probabilities)
#^^^ SAVE PROBABILITY HEATMAP TO FILE experiment 1


