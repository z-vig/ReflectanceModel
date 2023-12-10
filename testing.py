import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib import colormaps as cm

rfl_array = np.load('rfl_array.npy')
sim_array = np.load('sim_array.npy')
w_y = np.load('w_y.npy')
w_x = np.arange(0,1,0.02)

# fig,[ax1,ax2,ax3] = plt.subplots(1,3,figsize=(10,5),width_ratios=(1,1,5))
fig = plt.figure(layout="constrained")
gs = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[:,1:])
ax1.set_yticks([])
ax1.set_xticks([])
ax2.set_yticks([])
ax2.set_xticks([])
ax3.set_xlabel('Single Scattering Albedo')
ax3.set_ylabel('Reflectance')
art_list1 = []
art_list2 = []
art_list3 = []
for i in range(rfl_array.shape[2]):
    art1 = ax1.imshow(rfl_array[:,:,i],vmin=rfl_array.min(),vmax=rfl_array.max(),cmap='Reds')
    art2 = ax2.imshow(sim_array[:,:,i],vmin=0.25,vmax=0.83,cmap='Reds')
    cmap = cm['Reds']
    art3, = ax3.plot(w_x,w_y[i,:],color=cmap(i/rfl_array.shape[2]))
    art_list1.append([art1,art2,art3])
    # art_list2.append([art2])
    # art_list3.append([art3])

anim1 = ArtistAnimation(fig,art_list1,interval=50,blit=True,repeat=False)
# anim2 = ArtistAnimation(fig,art_list2,interval=50,blit=True,repeat=False)
# anim3 = ArtistAnimation(fig,art_list3,interval=50,blit=True,repeat=False)
anim1.save('timeseries.gif')
plt.show()