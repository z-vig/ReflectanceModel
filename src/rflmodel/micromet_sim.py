'''
This module simulates one round of space weathering via micrometeorite bombardment.
'''

import numpy as np
from tqdm import tqdm

class SimFrame:
    def __init__(self,frame_size:tuple,weather_rate:float,crater_rate:float,sim_length:int):
        self.fs = frame_size
        self.wr = weather_rate
        self.cr = crater_rate
        self.sl = sim_length
        self.frame = np.zeros(self.fs)
        self.anim_arr = np.zeros((*self.frame.shape,sim_length))
    
    def __str__(self) -> str:
        return f'{self.wr}_{self.cr}_{self.sl}'
    
    def __repr__(self) -> str:
        return f'SimFrame({self.wr},{self.cr},{self.sl})'

    def simstep(self):
        self.frame += self.wr

        x = np.arange(self.frame.shape[0])
        y = np.arange(self.frame.shape[1])
        r_range = np.arange(4,x.size)
        exp_dist = np.array([np.exp(-i) for i in r_range])

        for n in range(self.cr):
            cx = np.random.choice(x)
            cy = np.random.choice(y)
            r = np.random.choice(r_range,p=exp_dist/np.sum(exp_dist))
            self.frame[(x[np.newaxis,:]-cx)**2+(y[:,np.newaxis]-cy)**2 < r**2] = 0
    
    def runsim(self):
        for i in tqdm(range(self.sl)):
            self.anim_arr[:,:,i] = self.frame
            self.simstep()
        return self.anim_arr
        


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import ArtistAnimation

    xsize=250
    ysize=250
    test = SimFrame((xsize,ysize),0.01,1,100)
    animarr = test.runsim()

    art_list = []
    fig,ax = plt.subplots(1,1)
    for i in np.arange(animarr.shape[2]):
        im_art = ax.imshow(animarr[:,:,i],vmin=0,vmax=1,cmap='Reds')
        art_list.append([im_art])
    
    anim = ArtistAnimation(fig,art_list,interval=50,blit=True,repeat=False)

    plt.show()



    