import numpy as np
import pandas as pd
import pickle
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import scipy.ndimage

from numpy import *
from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter

def arc_patch(xy, width, height, theta1=0., theta2=180., resolution=50, **kwargs):

    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack((width*np.cos(theta)  + xy[0],
                        height*np.sin(theta) + xy[1]))
    # build the polygon and add it to the axes
    poly = matplotlib.patches.Polygon(points.T, closed=True, **kwargs)

    return poly

def draw_half_rink(fig, ax):
    # blue line
    x1, y1 = [25,25], [0, 84]
    ax.plot(x1,y1,'b')

    # goal line
    x1, y1 = [89, 89], [0, 84]
    ax.plot(x1,y1,'r')

    # curved boundry behind goal line
    w1 = matplotlib.patches.Wedge([72,56], 28, 0, 90, width=0.01, edgecolor='black', facecolor='none')
    w2 = matplotlib.patches.Wedge([72,28], 28, 270, 360, width=0.01, edgecolor='black', facecolor='none')
    ax.add_artist(w1)
    ax.add_artist(w2)

    # goal
    g = matplotlib.patches.Rectangle([89,39], 3.333, 6, angle=0.0)
    ax.add_artist(g)

    # faceoff circles
    fc1 = matplotlib.patches.Circle([69,20.5], 15, edgecolor='r',facecolor='none')
    fc2 = matplotlib.patches.Circle([69,63.5], 15, edgecolor='r',facecolor='none')
    # center faceoff circle
    fc3 = matplotlib.patches.Circle([0,42], 15, edgecolor='b',facecolor='none')
    ax.add_artist(fc1)
    ax.add_artist(fc2)
    ax.add_artist(fc3)

    # interior circles
    fc1 = matplotlib.patches.Circle([69,20.5], 1, edgecolor='r',facecolor='r')
    fc2 = matplotlib.patches.Circle([69,63.5], 1, edgecolor='r',facecolor='r')
    # red circles behind blue line
    fc3 = matplotlib.patches.Circle([20,20.5], 1, edgecolor='r',facecolor='r')
    fc4 = matplotlib.patches.Circle([20,63.5], 1, edgecolor='r',facecolor='r')
    # center faceoff
    fc5 = matplotlib.patches.Circle([0,42], 1, edgecolor='b',facecolor='b')
    ax.add_artist(fc1)
    ax.add_artist(fc2)
    ax.add_artist(fc3)
    ax.add_artist(fc4)
    ax.add_artist(fc5)

    # crease
    crease=arc_patch(xy=[89,42], width=4,
                    height=4, theta1=90, theta2=270, edgecolor='b',facecolor='b', alpha=.5)
    ax.add_artist(crease)
    return fig, ax




#end
