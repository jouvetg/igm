 
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

ss = np.loadtxt('data/EDC_dD_temp_estim.tab',dtype=np.float32,skiprows=31)
time = ss[:,1] * -1000  # extract time BP, chnage unit to yeat
dT   = ss[:,3]          # extract the dT, i.e. global temp. difference
dT =  interp1d(time,dT, fill_value=(dT[0], dT[-1]),bounds_error=False)
TIME = np.arange(-30000,-15000,100) # discretize time between 30 and 20 ka BP with century steps
 
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(TIME, dT(TIME),'-r') 
ax2.plot(TIME, 3000 + 200.0*dT(TIME),'b') 
ax1.set_xlabel('Time')
ax1.set_ylabel('DT', color='g')
ax2.set_ylabel('ELA', color='b')
plt.show()
