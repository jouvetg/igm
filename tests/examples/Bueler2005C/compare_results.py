import numpy as np
import netCDF4
import matplotlib.pyplot as plt

ncfile01 = netCDF4.Dataset('output.nc', mode='r')

IGM_icesurface = ncfile01['usurf'][10,:,:]
IGM_icesurface600 = ncfile01['usurf'][6,:,:]
IGM_smbfinal = ncfile01['smb'][10,:,:]

ncfile02 = netCDF4.Dataset('SIA_data_bueler.nc', mode='r')

SIA_icesurface = ncfile02['usurfSIA'][:,:]
Bueler_refsurf = ncfile02['usurfBueler'][:,:]

ncfile01.close()
ncfile02.close()

plt.figure()
plt.imshow(IGM_smbfinal, cmap="viridis")
plt.colorbar()
plt.title('Final SMB IGM (t=1000 y)')

plt.figure()
plt.imshow(IGM_icesurface, cmap="viridis")
plt.colorbar()
plt.title('Final IGM surface (t=1000 y)')

plt.figure()
plt.imshow(IGM_icesurface600, cmap="viridis")
plt.colorbar()
plt.title('IGM surface (t=600 y)')

plt.figure()
plt.imshow(Bueler_refsurf - IGM_icesurface, cmap='seismic')
plt.colorbar()
plt.clim(-50,50)
plt.title('usurf Bueler - IGM (t=1000 y)')

plt.figure()
plt.imshow(Bueler_refsurf - SIA_icesurface, cmap='seismic')
plt.colorbar()
plt.clim(-50,50)
plt.title('usurf Bueler - SIA FDM MUSCL superbee (t=1000 y)')

plt.show()
