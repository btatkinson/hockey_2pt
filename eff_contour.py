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
# uni_x = np.unique(df.X)
# uni_y = np.unique(df.Y)
#
# uni_x = np.sort(uni_x)
# uni_y = np.sort(uni_y)

# ONLY NEED TO DO ONCE THEN SAVE
# _X, _Y = np.meshgrid(uni_x,uni_y)

# rect_dict = {}

# for xrow in _X:
#     for _x in xrow:
#         rect_dict[_x] = {}
#
#         for yrow in _Y:
#             for _y in yrow:
#                 rect_dict[_x][_y]=0
#
# for index,row in tqdm(df.iterrows()):
#     _x = row['X']
#     _y = row['Y']
#     rect_dict[_x][_y] += 1
#
# _Z = pd.DataFrame.from_dict(rect_dict)
#
# print(_Z.head())

# print(_X.shape)
# print(_Y.shape)
# print(_Z.shape)

# with open ('_X.pkl', 'wb') as fp:
#     pickle.dump(_X,fp)
#     fp.close()
# with open ('_Y.pkl', 'wb') as fp:
#     pickle.dump(_Y,fp)
#     fp.close()
# with open ('_Z.pkl', 'wb') as fp:
#     pickle.dump(_Z,fp)
#     fp.close()

## END COMMENTED OUT
###################################


# PLOT SHOTS

# with open ('_X.pkl', 'rb') as fp:
#     _X = pickle.load(fp)
# with open ('_Y.pkl', 'rb') as fp:
#     _Y = pickle.load(fp)
# with open ('_Z.pkl', 'rb') as fp:
#     _Z = pickle.load(fp)

# sigma = 0.7 # smoothing value
# data = gaussian_filter(_Z, sigma)
#
# fig, ax = plt.subplots()
# CS = ax.contour(data)
# ax.clabel(CS, inline=1, fontsize=10)
# ax.set_title('Simplest default with labels')
#
# # blue line
# x1, y1 = [25,25], [0, 84]
# plt.plot(x1,y1,'b')
#
# # goal line
# x1, y1 = [89, 89], [0, 84]
# plt.plot(x1,y1,'r')
#
#
# plt.show()

# ONLY NEED TO DO ONCE FOR GOALS
# _X, _Y = np.meshgrid(uni_x,uni_y)
#
# goal_dict = {}
# goal_df = df.loc[df['Shot Result']=='GOAL']
# for xrow in _X:
#     for _x in xrow:
#         goal_dict[_x] = {}
#
#         for yrow in _Y:
#             for _y in yrow:
#                 goal_dict[_x][_y]=0
#
# for index,row in tqdm(goal_df.iterrows()):
#     _x = row['X']
#     _y = row['Y']
#     goal_dict[_x][_y] += 1
#
# goal_Z = pd.DataFrame.from_dict(goal_dict)
#
# with open ('_gX.pkl', 'wb') as fp:
#     pickle.dump(_X,fp)
#     fp.close()
# with open ('_gY.pkl', 'wb') as fp:
#     pickle.dump(_Y,fp)
#     fp.close()
# with open ('_gZ.pkl', 'wb') as fp:
#     pickle.dump(goal_Z,fp)
#     fp.close()

## END COMMENTED OUT
########################

# with open ('_X.pkl', 'rb') as fp:
#     _X = pickle.load(fp)
# with open ('_Y.pkl', 'rb') as fp:
#     _Y = pickle.load(fp)
with open ('_Z.pkl', 'rb') as fp:
    _Z = pickle.load(fp)
with open ('_gZ.pkl', 'rb') as fp:
    goal_Z = pickle.load(fp)

min_threshold = 5

# minimum sample size
mask = (_Z>min_threshold)
z_mask = goal_Z[mask]

efficien_Z = z_mask/_Z
max_eff = 0.24
efficien_Z[efficien_Z  > max_eff] = max_eff


x = np.arange(0, 100)
y = np.arange(0, 85)
arr = np.zeros((y.size, x.size))

# center of circle & radius
cx = 89
cy = 42
r = 16

# my 2 point line ellipse dimensions
# w=.70
# h=1.25
# mask1 = ((x[np.newaxis,:]-cx)**2)/(h**2) + ((y[:,np.newaxis]-cy)**2)/(w**2) < r**2
# mask = mask1
# two_Z = efficien_Z.copy()
# two_Z[~mask] = ma.dot(two_Z,2)

# resembles free throw line
# mask1 = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
# mask2 = (0 + y[:,np.newaxis]) > 34
# mask3 = (0 + y[:,np.newaxis]) < 50
# mask = mask1 & mask2 & mask3
# two_Z = efficien_Z
# two_Z[~mask] = ma.dot(two_Z,2)

# print(two_Z)
# print(two_Z.shape)

sigma = 0.6 # smoothing value
data1 = gaussian_filter(efficien_Z, sigma)
data2 = gaussian_filter(two_Z, sigma)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 2, 1)
CS = ax1.contourf(data1, 6)
CB = fig1.colorbar(CS)
# ax1.clabel(CS)

ax2 = fig1.add_subplot(1, 2, 2)
CS = ax2.contourf(data2, 6)
CB = fig1.colorbar(CS)
# ax2.clabel(CS)

def arc_patch(xy, width, height, theta1=0., theta2=180., resolution=50, **kwargs):

    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack((width*np.cos(theta)  + xy[0],
                        height*np.sin(theta) + xy[1]))
    # build the polygon and add it to the axes
    poly = matplotlib.patches.Polygon(points.T, closed=True, **kwargs)

    return poly

# 2 point arc

# my 2 point line ellipse
# a=arc_patch(xy=[89,42], width=19,
#                 height=11, theta1=90, theta2=270, edgecolor='orange',facecolor='none')
# ax2.add_artist(a)

# resembles free throw line
# a=arc_patch(xy=[89,42], width=16,
#                 height=16, theta1=150, theta2=210, edgecolor='orange',facecolor='none')
# ax.add_artist(a)
# x1, y1 = [75.5, 89], [50, 50]
# plt.plot(x1,y1,'orange',linewidth=0.8)
#
# x1, y1 = [75.5, 89], [34, 34]
# plt.plot(x1,y1,'orange',linewidth=0.8)




ax1.set_title('Current Efficiency Map')
ax2.set_title('2-Point Line Hypothetical Efficiency Map')

fig1, ax1 = draw_half_rink(fig1,ax1)
fig1, ax2 = draw_half_rink(fig1,ax2)

plt.show()



# end
