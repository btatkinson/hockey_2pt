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

min_threshold = 8


z_mask = goal_Z[_Z>min_threshold]

efficien_Z = z_mask/_Z
max_eff = 0.24
efficien_Z[efficien_Z  > max_eff] = max_eff

# efficien_Z = []
# for y,row in enumerate(_Z):
#     new_row = []
#     for x in row:
#         print(x,y)
#         z_val = int(x)
#         goal_val = int(goal_Z[y][x])
#         eZ = 0
#         if z_val >= min_threshold:
#             if goal_val >= 0:
#                 eZ = goal_Z/_Z
#         new_row.append(eZ)
#     efficien_Z.append(new_row)

sigma = 0.7 # smoothing value
data = gaussian_filter(efficien_Z, sigma)

fig, ax = plt.subplots()
CS = ax.contourf(data, 3)
CB = fig.colorbar(CS)
# ax.clabel(CS)

# Ellipse
# e = matplotlib.patches.Ellipse([89,42], width=35, height=22, angle=0,edgecolor='red',facecolor='none')
# ax.add_artist(e)

def arc_patch(xy, width, height, theta1=0., theta2=180., resolution=50, **kwargs):

    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack((width*np.cos(theta)  + xy[0],
                        height*np.sin(theta) + xy[1]))
    # build the polygon and add it to the axes
    poly = matplotlib.patches.Polygon(points.T, closed=True, **kwargs)

    return poly

a=arc_patch(xy=[89,42], width=16,
                height=11, theta1=90, theta2=270, edgecolor='orange',facecolor='none')
ax.add_artist(a)


ax.set_title('Hypothetical 2 point line')

# blue line
x1, y1 = [25,25], [0, 84]
plt.plot(x1,y1,'b')

# goal line
x1, y1 = [89, 89], [0, 84]
plt.plot(x1,y1,'r')

# goal
g = matplotlib.patches.Rectangle([89,39], 3, 6, angle=0.0)
ax.add_artist(g)

# faceoff circles
# 20.5
# 64.5
# 65 is a guess
fc1 = matplotlib.patches.Circle([65,20.5], 15, edgecolor='r',facecolor='none')
fc2 = matplotlib.patches.Circle([65,64.5], 15, edgecolor='r',facecolor='none')
ax.add_artist(fc1)
ax.add_artist(fc2)

# crease
crease=arc_patch(xy=[89,42], width=4,
                height=4, theta1=90, theta2=270,edgecolor='b',facecolor='none')
ax.add_artist(crease)

plt.show()





# end
