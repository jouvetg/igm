
from netCDF4 import Dataset
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import numpy as np
import sys

area = 'linth' # chose your area among lyon, linth, rhine, ticino

# import flowline and define distance along flowline
fl = np.loadtxt('data/'+area+'-flowline.dat')
dist = np.cumsum( np.sqrt( np.diff(fl[:,0])**2 + np.diff(fl[:,1])**2 ) )
dist = np.insert(dist,0,0) / 1000

# open IGM's output
nc      = Dataset('output.nc', 'r')        
time    = nc.variables['time'][:] ;
x       = np.squeeze(nc.variables['x']) ;
y       = np.squeeze(nc.variables['y']) ;
thk     = np.squeeze(nc.variables['thk']) ;
nc.close()

# check what points of the flowiline are covered by ice at time t. 
cov = np.zeros((len(time),len(fl)))
for it in range(len(time)):
    f = RectBivariateSpline(x, y, np.transpose(thk[it]))
    cov[it,:] = f(fl[:,0],fl[:,1],grid=False)>10

# 2D space position along flowline x times
DIST,TIME = np.meshgrid(dist,time)
D = DIST[cov==1]
T = TIME[cov==1]
 
# define the maximum ice thickness
thkmax = np.max(thk,axis=0)
thkmaxnan = np.where( thkmax<1 , np.nan, thkmax)

# plot flowline 
fig, (ax1, ax2) =  plt.subplots(2,1,figsize=(5,5),dpi=200) 
im1 = ax1.imshow(thkmaxnan, origin='lower',  extent=[min(x),max(x),min(y),max(y)])
ax1.plot(fl[:,0],fl[:,1],'-k', linewidth=2)
ax1.set_title('Maximum ice thickness with flowline')
ax1.axis('off') 
ax2.plot(D,T/1000, color='gray', alpha=0.75)
ax2.set(xlabel='Glacier position (km)', ylabel='Timing (ky)')
