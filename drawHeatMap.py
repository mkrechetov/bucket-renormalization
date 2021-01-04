from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

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

def drawProbabilityHeatmap(passedFileName,tractUVCoords,rawSeattleImage,initInfection,H_a,mu,TAU,probabilities):
    numPlots=0
    for i in initInfection:
        for h in H_a:
            for m in mu:
                for t in TAU:
                    numPlots=numPlots+1

    if numPlots<=5:
        fig, mainAxe = plt.subplots(figsize=(19.20, 4.6), constrained_layout=True)
        mainAxe.set_visible(False)
        if len(initInfection) > 1:
            fig.suptitle('Marginal Probabilities' + r' $\tau$' + '=' + t, fontsize=16)
        else:
            fig.suptitle('Marginal Probabilities initial infection=[' + initInfection[0] + ']' + r' $\tau$' + '=' + t, fontsize=16)
        gs = GridSpec(1, numPlots, figure=fig)

        c = mcolors.ColorConverter().to_rgb
        rvb = make_colormap(
            [c('blue'), c('red')])


        counter = 0
        #axes=[]
        for i in initInfection:
            for h in H_a:
                for m in mu:
                    for t in TAU:
                        ax = fig.add_subplot(gs[0, counter])
                        #ax.axes.xaxis.set_visible(False)
                        #ax.axes.yaxis.set_visible(False)
                        ax.set_title('Initial infection='+i+r' $H_a$'+'=' + h + r' $\mu$=' + m,fontsize=8)
                        #axes.append(ax)

                        ax.imshow(rawSeattleImage, interpolation="nearest")

                        # inset axes....
                        axins = ax.inset_axes([0.01, 0.5, 0.47, 0.47])
                        axins.imshow(rawSeattleImage, interpolation="nearest",origin="lower")
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
                            if str(index) in initInfection:
                                actualPoint=[tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]]
                                Size=12
                                path = [
                                    [actualPoint[0], actualPoint[1] - Size],
                                    [actualPoint[0] - Size * 0.3, actualPoint[1] - Size * 0.3],
                                    [actualPoint[0] - Size, actualPoint[1] - Size * 0.1],
                                    [actualPoint[0] - Size * 0.2, actualPoint[1] + Size * 0.1],
                                    [actualPoint[0] - Size * 0.8, actualPoint[1] + Size * 0.9],
                                    [actualPoint[0], actualPoint[1] + Size * 0.6],
                                    [actualPoint[0] + Size * 0.8, actualPoint[1] + Size * 0.9],
                                    [actualPoint[0] + Size * 0.2, actualPoint[1] + Size * 0.1],
                                    [actualPoint[0] + Size, actualPoint[1] - Size * 0.1],
                                    [actualPoint[0] + Size * 0.3, actualPoint[1] - Size * 0.3],
                                ]
                                ax.add_patch(Polygon(path,
                                            color=(1,0,0, 1),
                                            linewidth=1))
                            else:
                                ax.add_patch(
                                    Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=11,
                                            height=11,
                                            edgecolor='None',
                                            facecolor=(probabilities[counter].iloc[0][index], 0,
                                                       1 - probabilities[counter].iloc[0][index], 1),
                                            linewidth=1))

                        for index in range(probabilities[counter].shape[1]):
                            if str(index) in initInfection:
                                actualPoint = [tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]]
                                Size = 14
                                path = [
                                    [actualPoint[0], actualPoint[1] - Size],
                                    [actualPoint[0] - Size * 0.3, actualPoint[1] - Size * 0.3],
                                    [actualPoint[0] - Size, actualPoint[1] - Size * 0.1],
                                    [actualPoint[0] - Size * 0.2, actualPoint[1] + Size * 0.1],
                                    [actualPoint[0] - Size * 0.8, actualPoint[1] + Size * 0.9],
                                    [actualPoint[0], actualPoint[1] + Size * 0.6],
                                    [actualPoint[0] + Size * 0.8, actualPoint[1] + Size * 0.9],
                                    [actualPoint[0] + Size * 0.2, actualPoint[1] + Size * 0.1],
                                    [actualPoint[0] + Size, actualPoint[1] - Size * 0.1],
                                    [actualPoint[0] + Size * 0.3, actualPoint[1] - Size * 0.3],
                                ]
                                ax.add_patch(Polygon(path,
                                                     color=(1, 0, 0, 1),
                                                     linewidth=1))
                            else:
                                axins.add_patch(
                                    Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=15,
                                            height=15,
                                            edgecolor='None',
                                            facecolor=(probabilities[counter].iloc[0][index], 0,
                                                       1 - probabilities[counter].iloc[0][index], 1),
                                            linewidth=1))

                        counter = counter+1

    elif numPlots>5 and numPlots<=12:
        fig, mainAxe = plt.subplots(figsize=(19.20, 10), constrained_layout=True)
        mainAxe.set_visible(False)
        if len(initInfection) > 1:
            fig.suptitle('Marginal Probabilities' + r' $\tau$' + '=' + t, fontsize=16)
        else:
            fig.suptitle('Marginal Probabilities initial infection=[' + initInfection[0] + ']' + r' $\tau$' + '=' + t, fontsize=16)
        maxRows = 3
        maxColumn=4
        gs = GridSpec(math.ceil(numPlots/maxColumn), maxColumn, figure=fig)

        c = mcolors.ColorConverter().to_rgb
        rvb = make_colormap(
            [c('blue'), c('red')])


        counterP=0
        counterR = 0
        counterC=0
        # axes=[]
        for i in initInfection:
            for h in H_a:
                for m in mu:
                    for t in TAU:
                        print('___')
                        print(counterR)
                        print(counterC)
                        print('___')
                        ax = fig.add_subplot(gs[counterR, counterC])
                        ax.axes.xaxis.set_visible(False)
                        ax.axes.yaxis.set_visible(False)
                        ax.set_title('Initial infection='+i+r' $H_a$'+'=' + h + r' $\mu$=' + m,fontsize=8)
                        # axes.append(ax)

                        ax.imshow(rawSeattleImage, interpolation="nearest")

                        # inset axes....
                        axins = ax.inset_axes([0.01, 0.5, 0.47, 0.47])
                        axins.imshow(rawSeattleImage, interpolation="nearest",origin="lower")
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
                            if str(index) in initInfection:
                                actualPoint = [tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]]
                                Size = 12
                                path = [
                                    [actualPoint[0], actualPoint[1] - Size],
                                    [actualPoint[0] - Size * 0.3, actualPoint[1] - Size * 0.3],
                                    [actualPoint[0] - Size, actualPoint[1] - Size * 0.1],
                                    [actualPoint[0] - Size * 0.2, actualPoint[1] + Size * 0.1],
                                    [actualPoint[0] - Size * 0.8, actualPoint[1] + Size * 0.9],
                                    [actualPoint[0], actualPoint[1] + Size * 0.6],
                                    [actualPoint[0] + Size * 0.8, actualPoint[1] + Size * 0.9],
                                    [actualPoint[0] + Size * 0.2, actualPoint[1] + Size * 0.1],
                                    [actualPoint[0] + Size, actualPoint[1] - Size * 0.1],
                                    [actualPoint[0] + Size * 0.3, actualPoint[1] - Size * 0.3],
                                ]
                                ax.add_patch(Polygon(path,
                                                     color=(1, 0, 0, 1),
                                                     linewidth=1))
                            else:
                                ax.add_patch(
                                    Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=11,
                                            height=11,
                                            edgecolor='None',
                                            facecolor=(
                                                probabilities[counterP].iloc[0][index], 0,
                                                1 - probabilities[counterP].iloc[0][index],
                                                1),
                                            linewidth=1))
                        for index in range(probabilities[counterP].shape[1]):
                            if str(index) in initInfection:
                                actualPoint = [tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]]
                                Size = 14
                                path = [
                                    [actualPoint[0], actualPoint[1] - Size],
                                    [actualPoint[0] - Size * 0.3, actualPoint[1] - Size * 0.3],
                                    [actualPoint[0] - Size, actualPoint[1] - Size * 0.1],
                                    [actualPoint[0] - Size * 0.2, actualPoint[1] + Size * 0.1],
                                    [actualPoint[0] - Size * 0.8, actualPoint[1] + Size * 0.9],
                                    [actualPoint[0], actualPoint[1] + Size * 0.6],
                                    [actualPoint[0] + Size * 0.8, actualPoint[1] + Size * 0.9],
                                    [actualPoint[0] + Size * 0.2, actualPoint[1] + Size * 0.1],
                                    [actualPoint[0] + Size, actualPoint[1] - Size * 0.1],
                                    [actualPoint[0] + Size * 0.3, actualPoint[1] - Size * 0.3],
                                ]
                                ax.add_patch(Polygon(path,
                                                     color=(1, 0, 0, 1),
                                                     linewidth=1))
                            else:
                                axins.add_patch(
                                    Ellipse((tractUVCoords.iloc[index][1], tractUVCoords.iloc[index][2]), width=15,
                                            height=15,
                                            edgecolor='None',
                                            facecolor=(probabilities[counterP].iloc[0][index], 0,
                                                       1 - probabilities[counterP].iloc[0][index], 1),
                                            linewidth=1))


                        counterR=counterR+1
                        if counterR>=maxRows:
                            counterC=counterC+1
                            counterR=0

                        counterP = counterP + 1
    else:
        print('not implemented')

    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    divider = make_axes_locatable(mainAxe)
    cax = divider.append_axes('right', size='1%', pad='1%')

    # cax1 = cax.append_axes("right", size="7%", pad="2%")
    # cb1 = colorbar(im1, cax=cax1)

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


tractUVCoords = pd.read_csv('./seattle/tractUVCoordinates.csv')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES
rawSeattleImage=mpimg.imread('./seattle/SeattleRawImage1.png')#GIS DATA WHICH SHOULD BE READ ONCE AND USED MULTIPLE TIMES


# initInfection=['52']
# H_a=['0.0']
# mu=['0.001']
# tau=['120.0']
# probabilities=[]
# for i in initInfection:
#     for h in H_a:
#         for m in mu:
#             for t in tau:
#                 file_name='./results/seattle_marg_prob_init_inf=['+i+']_H_a='+h+'_MU='+m+'_TAU='+t+'.csv'
#                 temp=pd.read_csv(file_name)
#                 temp = renormalizeProbability(temp)
#                 probabilities.append(temp)
#
# test_name = "./results/seattle_heatmap_init_inf=[0]_H_a=[0.0,0.1,0.5]_MU_[0.001,0.0015,0.002]_TAU_[120]"
# drawProbabilityHeatmap(test_name,tractUVCoords,rawSeattleImage,initInfection,H_a,mu,tau,probabilities)


initInfection=['0']
H_a=['0.0','0.1','0.5','1.0']
mu=['0.001','0.0015','0.002']
tau=['120.0']
probabilities=[]
for i in initInfection:
    for h in H_a:
        for m in mu:
            for t in tau:
                file_name='./results/seattle_marg_prob_init_inf=['+i+']_H_a='+h+'_MU='+m+'_TAU='+t+'.csv'
                temp=pd.read_csv(file_name)
                temp = renormalizeProbability(temp)
                probabilities.append(temp)

test_name = "./results/seattle_heatmap_init_inf=[0]_H_a=[0.0,0.1,0.5]_MU_[0.001,0.0015,0.002]_TAU_[120]"
drawProbabilityHeatmap(test_name,tractUVCoords,rawSeattleImage,initInfection,H_a,mu,tau,probabilities)

initInfection=['52']
H_a=['0.0','0.1','0.5','1.0']
mu=['0.001','0.0015','0.002']
tau=['120.0']
probabilities=[]
for i in initInfection:
    for h in H_a:
        for m in mu:
            for t in tau:
                file_name='./results/seattle_marg_prob_init_inf=['+i+']_H_a='+h+'_MU='+m+'_TAU='+t+'.csv'
                temp=pd.read_csv(file_name)
                temp = renormalizeProbability(temp)
                probabilities.append(temp)

test_name = "./results/seattle_heatmap_init_inf=[52]_H_a=[0.0,0.1,0.5]_MU_[0.001,0.0015,0.002]_TAU_[120]"
drawProbabilityHeatmap(test_name,tractUVCoords,rawSeattleImage,initInfection,H_a,mu,tau,probabilities)