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
from helpers import draw_half_rink

df = pd.read_pickle("shot_locations.pkl")

# 685,509 shots

# blocked shots apparently have the coordinates of the block?

# transfer all locations to same half
df['X'] = df.copy().X.abs()

df=df.round(5)

# get a count of all events at every point

# contour plor requires a rectangular grid
uni_x = np.unique(df.X)
uni_y = np.unique(df.Y)

uni_x = np.sort(uni_x)
uni_y = np.sort(uni_y)

def arc_patch(xy, width, height, theta1=0., theta2=180., resolution=50, **kwargs):

    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack((width*np.cos(theta)  + xy[0],
                        height*np.sin(theta) + xy[1]))
    # build the polygon and add it to the axes
    poly = matplotlib.patches.Polygon(points.T, closed=True, **kwargs)

    return poly

# ONLY NEED TO DO ONCE THEN SAVE
def generate_shot_input(uni_x, uni_y):
    _X, _Y = np.meshgrid(uni_x,uni_y)

    rect_dict = {}

    for xrow in _X:
        for _x in xrow:
            rect_dict[_x] = {}

            for yrow in _Y:
                for _y in yrow:
                    rect_dict[_x][_y]=0

    for index,row in tqdm(df.iterrows()):
        _x = row['X']
        _y = row['Y']
        rect_dict[_x][_y] += 1

    _Z = pd.DataFrame.from_dict(rect_dict)
    with open ('_X.pkl', 'wb') as fp:
        pickle.dump(_X,fp)
        fp.close()
    with open ('_Y.pkl', 'wb') as fp:
        pickle.dump(_Y,fp)
        fp.close()
    with open ('_Z.pkl', 'wb') as fp:
        pickle.dump(_Z,fp)
        fp.close()
    return

# ONLY NEED TO DO ONCE FOR GOALS
def generate_goal_input(uni_x,uni_y,df):
    _X, _Y = np.meshgrid(uni_x,uni_y)
    goal_dict = {}
    goal_df = df.loc[df['Shot Result']=='GOAL']
    for xrow in _X:
        for _x in xrow:
            goal_dict[_x] = {}
            for yrow in _Y:
                for _y in yrow:
                    goal_dict[_x][_y]=0
    for index,row in tqdm(goal_df.iterrows()):
        _x = row['X']
        _y = row['Y']
        goal_dict[_x][_y] += 1

    goal_Z = pd.DataFrame.from_dict(goal_dict)

    with open ('_gX.pkl', 'wb') as fp:
        pickle.dump(_X,fp)
        fp.close()
    with open ('_gY.pkl', 'wb') as fp:
        pickle.dump(_Y,fp)
        fp.close()
    with open ('_gZ.pkl', 'wb') as fp:
        pickle.dump(goal_Z,fp)
        fp.close()
    return


# only need to do the functions below once
# generate_shot_input(uni_x,uni_y)
# generate_goal_input(uni_x,uni_y,df)



## END COMMENTED OUT
###################################


# PLOT Contours

def plot_contours():
    with open ('_X.pkl', 'rb') as fp:
        _X = pickle.load(fp)
    with open ('_Y.pkl', 'rb') as fp:
        _Y = pickle.load(fp)
    with open ('_Z.pkl', 'rb') as fp:
        _Z = pickle.load(fp)

    sigma = 0.7 # smoothing value
    data = gaussian_filter(_Z, sigma)

    fig, ax = plt.subplots()
    CS = ax.contour(data)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Simplest default with labels')
    fig, ax = draw_half_rink(fig, ax)
    plt.show()
    return

def plot_2p():
    with open ('_X.pkl', 'rb') as fp:
        _X = pickle.load(fp)
    with open ('_Y.pkl', 'rb') as fp:
        _Y = pickle.load(fp)
    with open ('_Z.pkl', 'rb') as fp:
        _Z = pickle.load(fp)
    with open ('_gZ.pkl', 'rb') as fp:
        goal_Z = pickle.load(fp)

    # minimum sample size
    min_threshold = 5
    mask = (_Z>min_threshold)
    z_mask = goal_Z[mask]

    efficien_Z = z_mask/_Z

    # maximize possible values
    max_eff = 0.24
    efficien_Z[efficien_Z  > max_eff] = max_eff

    # dummy points for mask
    x = np.arange(0, 100)
    y = np.arange(0, 85)
    arr = np.zeros((y.size, x.size))

    # center of circle & radius
    cx = 89
    cy = 42
    r = 16

    # my 2 point line ellipse dimensions
    w=1.4
    h=2.5
    mask1 = ((x[np.newaxis,:]-cx)**2)/(h**2) + ((y[:,np.newaxis]-cy)**2)/(w**2) < r**2
    mask = mask1
    two_Z = efficien_Z.copy()
    two_Z[~mask] = ma.dot(two_Z,2)

    sigma = 0.6 # smoothing value
    data = gaussian_filter(two_Z, sigma)

    fig1, ax1 = plt.subplots()
    CS = ax1.contourf(data, 6)
    CB = fig1.colorbar(CS)

    # 2 point arc
    # my 2 point line ellipse
    a=arc_patch(xy=[89,42], width=38.5,
                    height=22.5, theta1=90, theta2=270, edgecolor='orange',facecolor='none')
    ax1.add_artist(a)

    fig1, ax1 = draw_half_rink(fig1, ax1)
    plt.show()
    return

def plot_4():
    with open ('_X.pkl', 'rb') as fp:
        _X = pickle.load(fp)
    with open ('_Y.pkl', 'rb') as fp:
        _Y = pickle.load(fp)
    with open ('_Z.pkl', 'rb') as fp:
        _Z = pickle.load(fp)
    with open ('_gZ.pkl', 'rb') as fp:
        goal_Z = pickle.load(fp)

    min_threshold = 5

    # minimum sample size
    mask = (_Z>min_threshold)
    z_mask = goal_Z[mask]

    efficien_Z = z_mask/_Z

    # maximize possible values
    max_eff = 0.24
    efficien_Z[efficien_Z  > max_eff] = max_eff

    # dummy points for mask
    x = np.arange(0, 100)
    y = np.arange(0, 85)
    arr = np.zeros((y.size, x.size))

    # center of circle & radius
    cx = 89
    cy = 42
    r = 16

    # more like basketball
    w=.70
    h=1.25
    mask = ((x[np.newaxis,:]-cx)**2)/(h**2) + ((y[:,np.newaxis]-cy)**2)/(w**2) < r**2
    b_Z = efficien_Z.copy()
    b_Z[~mask] = ma.dot(b_Z,2)

    # single straight line
    mask = ((y[:,np.newaxis]*0) + x[np.newaxis,:]) > 52
    sl_Z = efficien_Z.copy()
    sl_Z[~mask] = ma.dot(sl_Z,2)

    # semi-circle
    _r = 29
    mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < _r**2
    # mask2 = (0 + y[:,np.newaxis]) > 34
    # mask3 = (0 + y[:,np.newaxis]) < 50
    # mask = mask1 & mask2 & mask3
    sc_Z = efficien_Z.copy()
    sc_Z[~mask] = ma.dot(sc_Z,2)

    # two point line and three point line
    w=.9333
    h=1.6666
    mask1 = ((x[np.newaxis,:]-cx)**2)/(h**2) + ((y[:,np.newaxis]-cy)**2)/(w**2) < r**2
    mask2 = ((x[np.newaxis,:]-cx)**2)/(h**2) + ((y[:,np.newaxis]-cy)**2)/(w**2) < r**2
    mask = mask1 & mask2
    two_Z = efficien_Z.copy()
    two_Z[~mask] = ma.dot(two_Z,2)
    w=1.86666
    h=3.3333
    mask = ((x[np.newaxis,:]-cx)**2)/(h**2) + ((y[:,np.newaxis]-cy)**2)/(w**2) < r**2
    three_Z = two_Z.copy()
    three_Z[~mask] = ma.dot(three_Z,1.5)
    # tiny value outside 3 point line that has .28 efficiency, i get rid of it
    max_eff = 0.24
    three_Z[three_Z  > max_eff] = max_eff

    sigma = 0.6 # smoothing value
    data1 = gaussian_filter(b_Z, sigma)
    data2 = gaussian_filter(sl_Z, sigma)
    data3 = gaussian_filter(sc_Z, sigma)
    data4 = gaussian_filter(three_Z, sigma)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(2, 2, 1)
    CS = ax1.contourf(data1, 6)
    CB = fig1.colorbar(CS)

    ax2 = fig1.add_subplot(2, 2, 2)
    CS = ax2.contourf(data2, 6)
    CB = fig1.colorbar(CS)

    ax3 = fig1.add_subplot(2, 2, 3)
    CS = ax3.contourf(data3, 6)
    CB = fig1.colorbar(CS)

    ax4 = fig1.add_subplot(2, 2, 4)
    CS = ax4.contourf(data4, 6)
    CB = fig1.colorbar(CS)

    # more like basketball
    a=arc_patch(xy=[89,42], width=19,
                    height=11, theta1=90, theta2=270, edgecolor='orange',facecolor='none')
    ax1.add_artist(a)

    # straight line
    x1, y1 = [52, 52], [0, 84]
    ax2.plot(x1,y1,'orange',linewidth=0.8)

    # resembles free throw line
    a=arc_patch(xy=[89,42], width=29,
                    height=29, theta1=90, theta2=270, edgecolor='orange',facecolor='none')
    ax3.add_artist(a)
    # x1, y1 = [75.5, 89], [50, 50]
    # ax3.plot(x1,y1,'orange',linewidth=0.8)
    #
    # x1, y1 = [75.5, 89], [34, 34]
    # ax3.plot(x1,y1,'orange',linewidth=0.8)

    # three point line
    a=arc_patch(xy=[89,42], width=25.666,
                    height=15, theta1=90, theta2=270, edgecolor='orange',facecolor='none')
    ax4.add_artist(a)
    a=arc_patch(xy=[89,42], width=52.6667,
                    height=30, theta1=90, theta2=270, edgecolor='y',facecolor='none')
    ax4.add_artist(a)


    ax1.set_title('Closer 2-Point Line')
    ax2.set_title('Straight Line')
    ax3.set_title('Semi-Circle')
    ax4.set_title('3 pt line :O')

    fig1, ax1 = draw_half_rink(fig1,ax1)
    fig1, ax2 = draw_half_rink(fig1,ax2)
    fig1, ax3 = draw_half_rink(fig1,ax3)
    fig1, ax4 = draw_half_rink(fig1,ax4)

    plt.show()

    return


# plot_contours()

plot_2p()

plot_4()





# end
