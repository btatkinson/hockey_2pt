import numpy as np
import pandas as pd
import pickle
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

from numpy import *

from scipy.stats import gaussian_kde

# delta = 0.025
# x = np.arange(-3.0, 3.0, delta)
# y = np.arange(-2.0, 2.0, delta)
# X, Y = np.meshgrid(x, y)
# print(X)
# print(Y)
# Z1 = np.exp(-X**2 - Y**2)
# print(Z1)
# Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# Z = (Z1 - Z2) * 2
#
# print(Z)
#
# raise ValueError

df = pd.read_pickle("shot_locations.pkl")

# 685,509 shots

# blocked shots apparently have the coordinates of the block?

# distance from goal
# only have to run once
# df['DistFG'] = df.apply(lambda x: math.sqrt(((89-abs(x['X']))**2) + (x['Y']**2)), axis=1)
# # save so don't have to keep doing operation
# df.to_pickle("shot_locations.pkl")

# transfer all locations to same half
df['X'] = df.copy().X.abs()

goal_df = df.loc[df['Shot Result']=='GOAL']
# ms_df = shot_df.loc[shot_df['Shot Result']!='GOAL']

## PLOT DISTANCE FROM GOAL
# z = goal_df.DistFG
# cmap = matplotlib.cm.get_cmap('viridis_r')
# normalize = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
# colors = [cmap(normalize(value)) for value in z]
#
# plt.scatter(x=goal_df.X.values, y=goal_df.Y.values, c=colors, s=4)
# plt.show()

# GOALS HEATMAP
# x = goal_df.X.values
# y = goal_df.Y.values
#
# xy = np.vstack([x,y])
# z = gaussian_kde(xy)(xy)
# idx = z.argsort()
# x, y, z = x[idx], y[idx], z[idx]
#
# fig, ax = plt.subplots()
# ax.scatter(x, y, c=z, s=10, edgecolor='')
# plt.show()

# HEXAGONAL BINS
# x = df.X
# y = df.Y
#
# xmin = x.min()
# xmax = x.max()
# ymin = y.min()
# ymax = y.max()
#
# fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(10, 6))
# fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
# ax = axs[0]
# hb = ax.hexbin(x, y, gridsize=25, cmap='inferno')
# ax.axis([xmin, xmax, ymin, ymax])
# ax.set_title("Hexagon binning")
# cb = fig.colorbar(hb, ax=ax)
# cb.set_label('counts')
#
# ax = axs[1]
# hb = ax.hexbin(x, y, gridsize=25, bins='log', cmap='inferno')
# ax.axis([xmin, xmax, ymin, ymax])
# ax.set_title("With a log color scale")
# cb = fig.colorbar(hb, ax=ax)
# cb.set_label('log10(N)')
#
# plt.show()


# Efficiency Map

print(df.head())

total_x = df.X
total_y = df.Y

xmin = total_x.min()
xmax = total_x.max()
ymin = total_y.min()
ymax = total_y.max()

thb = plt.hexbin(total_x, total_y, gridsize=35, cmap='inferno')

# tbin_xy = thb.get_offsets()
tbin_counts = thb.get_array()

goal_df = df.loc[df['Shot Result']=='GOAL']

goal_x = goal_df.X
goal_y = goal_df.Y

ghb = plt.hexbin(goal_x, goal_y, gridsize=35, cmap='inferno')

# gbin_xy = ghb.get_offsets()
gbin_counts = ghb.get_array()

# min threshold
gbin_counts[tbin_counts<=10] = 0

ehb=plt.hexbin(goal_x, goal_y, gridsize=35, vmin=0, cmap='inferno')

efficiency = np.divide(gbin_counts,tbin_counts)
sort = np.sort(efficiency)
efficiency[efficiency == -inf] = 0
efficiency[efficiency == inf] = 0
NaNs = isnan(efficiency)
efficiency[NaNs] = 0

x = np.linspace(0,10)
y = x**2

# blue line
x1, y1 = [25,25], [-42, 42]
plt.plot(x1,y1,'b')

# goal line
x1, y1 = [89, 89], [-42, 42]
plt.plot(x1,y1,'r')

# goal Rectangle patch
# goal = matplotlib.patches.Rectangle((89,1),3,2,linewidth=1,edgecolor='r',facecolor='none')

#
# sort = np.sort(efficiency)
# ehb.set_array(efficiency)

# efficiency coordinates and z values
# eff_xy = ehb.get_offsets()
# x = eff_xy[:,0]
# y = eff_xy[:,1]
# z = ehb.get_array()

# point_df = pd.DataFrame({'X':x,'Y':y, 'Z':z})
# point_df = point_df.round(5)

# MUST ALTER POINT LISTS TO BE A RECTANGULAR GRID
# X = np.unique(point_df.X)
# Y = np.unique(point_df.Y)
#
# _X,_Y = np.meshgrid(X,Y)

# initialize dict for lookup
# point_dict = {}
# for xrow in _X:
#     for _x in xrow:
#         point_dict[_x] = {}
#         # _x is x value of cell
#         for yrow in _Y:
#             for _y in yrow:
#                 point_dict[_x][_y] = 0
#
# max_value = 0.5
# populate dict with calculated values
# for idx, _x in enumerate(x):
#     x_val = x[idx]
#     y_val = y[idx]
#     x_val = np.round(x_val,5)
#     y_val = np.round(y_val,5)
#     _z = max(z[idx],max_value)
#     point_dict[x_val][y_val] = _z
#
# print(len(point_dict))
# print(len(point_dict[0]))

# make rectangular grid by looking up values

# grid = pd.DataFrame.from_dict(point_dict)
# print(grid.head())

# print(_X.shape)
# print(_Y.shape)
# print(grid.shape)
#
# fig, ax = plt.subplots()
# CS = ax.contour(_X, _Y, grid)
# ax.clabel(CS, inline=1, fontsize=10)
# ax.set_title('Simplest default with labels')

plt.show()







# end
