import numpy as np
import pandas as pd
import pickle
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

from numpy import *
from helpers import draw_half_rink
from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter

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

# make all positive
df['Y'] = df['Y'] + 42

goal_df = df.loc[df['Shot Result']=='GOAL']
# ms_df = shot_df.loc[shot_df['Shot Result']!='GOAL']

## PLOT DISTANCE FROM GOAL
# z = df.DistFG
# cmap = matplotlib.cm.get_cmap('viridis_r')
# normalize = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
# colors = [cmap(normalize(value)) for value in z]
# #
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(1, 2, 1)
# ax1.scatter(x=df.X.values, y=df.Y.values, c=colors, s=10, edgecolor='')
# fig1,ax1 = draw_half_rink(fig1,ax1)
# ax1.set_title('NHL Shot Locations 2014-2019')
#
# # GOALS HEATMAP
# x = goal_df.X.values
# y = goal_df.Y.values
#
# xy = np.vstack([x,y])
# z = gaussian_kde(xy)(xy)
# idx = z.argsort()
# x, y, z = x[idx], y[idx], z[idx]
#
# ax2 = fig1.add_subplot(1, 2, 2)
# ax2.set_title('NHL Goal Locations 2014-2019')
# ax2.scatter(x, y, c=z, cmap='magma',s=10, edgecolor='')
# fig1,ax2 = draw_half_rink(fig1,ax2)
# plt.show()

# HEXAGONAL BINS
x = df.X
y = df.Y

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()

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


# Efficiency Map

# fig1 = plt.figure()
# ax1 = fig1.add_subplot(1, 2, 1)
#
# total_x = df.X
# total_y = df.Y
#
# xmin = total_x.min()
# xmax = total_x.max()
# ymin = total_y.min()
# ymax = total_y.max()
#
# thb = ax1.hexbin(total_x, total_y, gridsize=35, cmap='inferno')
#
# # tbin_xy = thb.get_offsets()
# tbin_counts = thb.get_array()
#
# goal_df = df.loc[df['Shot Result']=='GOAL']
#
# goal_x = goal_df.X
# goal_y = goal_df.Y
#
# ghb = ax1.hexbin(goal_x, goal_y, gridsize=35, cmap='inferno')
#
# # gbin_xy = ghb.get_offsets()
# gbin_counts = ghb.get_array()
#
# # min threshold
# min_threshold = 10
# gbin_counts[tbin_counts<=min_threshold] = 0
#
# ehb=ax1.hexbin(goal_x, goal_y, gridsize=35, vmin=0, cmap='inferno')
#
# tbin_counts[tbin_counts<=0] = 0.001
# eff_arr = np.divide(gbin_counts,tbin_counts)
# eff_arr[eff_arr>0.3] = 0
# ehb.set_array(eff_arr)
# CB = fig1.colorbar(ehb)
#
# fig1,ax1 = draw_half_rink(fig1,ax1)

###
#######################

with open ('_Z.pkl', 'rb') as fp:
    _Z = pickle.load(fp)
with open ('_gZ.pkl', 'rb') as fp:
    goal_Z = pickle.load(fp)

min_threshold = 8


z_mask = goal_Z[_Z>min_threshold]

efficien_Z = z_mask/_Z
max_eff = 0.24
efficien_Z[efficien_Z  > max_eff] = max_eff

sigma = 0.7 # smoothing value
data = gaussian_filter(efficien_Z, sigma)

fig1,ax2 = plt.subplots()
# ax2 = fig1.add_subplot()
CS = ax2.contourf(data, 3)
CB = fig1.colorbar(CS)
fig1,ax2 = draw_half_rink(fig1,ax2)
ax2.set_title('Shooting Pct of Shots 2014-2019, Smoothed With Gaussian Filter')

plt.show()







# end
