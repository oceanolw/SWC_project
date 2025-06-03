import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sys
import cmocean
import xarray as xr
import xroms
import glob
import os
import datetime
import numpy as np
import dateutil


data_path = '/southern/rbarkan/data/SWC2km/OUTPUT/W_rivers/HIS/'
save_path = '/meddy/lwang/data/SWC2km_Wrivers_surface_uv/'
figure_dir = '/meddy/lwang/figures/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

filenames = sorted(glob.glob(data_path+'z_SWC2km_his.*.nc')) 
print('Number of files processed: %d' %len(filenames))

grd = xr.open_dataset('/southern/rbarkan/data/SWC2km/SWC2km_grd.nc')
ds = xr.open_mfdataset(filenames, concat_dim='time', combine="nested")

ds_uv = ds[['u','v']].isel(depth=0)

def u2rho_3d(var_u):

    [N,M,Lp]=var_u.shape
    Mp=M+1
    Mm=M-1
    var_rho=np.zeros((N,Mp,Lp))
    var_rho[:,1:M,:]=0.5*(var_u[:,0:Mm,:]+var_u[:,1:M,:])
    var_rho[:,0,:]=var_rho[:,1,:]
    var_rho[:,Mp-1,:]=var_rho[:,M-1,:]

    return var_rho

def v2rho_3d(var_v):

    [N,Mp,L]=var_v.shape
    Lp=L+1
    Lm=L-1
    var_rho=np.zeros((N,Mp,Lp))
    var_rho[:,:,1:L]=0.5*(var_v[:,:,0:Lm]+var_v[:,:,1:L,])
    var_rho[:,:,0]=var_rho[:,:,1]
    var_rho[:,:,Lp-1]=var_rho[:,:,L-1]
    return var_rho

u = v2rho_3d(ds_uv['u'])
v = u2rho_3d(ds_uv['v'])
u_bar = np.mean(u,axis=0)
v_bar = np.mean(v,axis=0)

eke = 0.5* ( (u - u_bar)**2 + (v - v_bar)**2 )
eke = np.where(eke==0, np.nan, eke)

#################################################################################
fig, ax = plt.subplots(1,1,figsize=(9.,6.))
m = Basemap(projection='merc',llcrnrlat=0,urcrnrlat=27,\
            llcrnrlon=97,urcrnrlon=132,lat_ts=20,resolution='i')
m.drawcoastlines(linewidth=0.6)
m.fillcontinents(color='lightgrey');
m.drawparallels(np.arange(0,30,5),labels=[1,0,0,0])
m.drawmeridians(np.arange(97,137,8),labels=[0,0,0,1])

xp, yp = m(grd['lon_rho'], grd['lat_rho'])
plot = m.pcolormesh(xp, yp, np.nanmean(eke,axis=0),vmin=0, vmax=0.2, cmap='Spectral_r')
cbar = fig.colorbar(plot,ax=ax,extend='both')
cbar.ax.set_ylabel('Kinetic energy [m' + r'$^2$' + ' s' + r'$^{-2}$' + ']', fontsize=14)

plt.savefig((figure_dir +'eke_annualmean.png', dpi=600, bbox_inches='tight'))