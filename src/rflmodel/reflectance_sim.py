'''
Running the simulation and experimenting with reflectance model
'''

from micromet_sim import SimFrame
from reflectance_calc import HapkeFrame

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy.stats import multivariate_normal
import numpy as np
from tqdm import tqdm


def get_instrument_response(arr_size:int):
    x,y = np.meshgrid(np.arange(0,arr_size,1),np.arange(0,arr_size,1))
    pos = np.dstack((x,y))
    means = np.array([arr_size/2,arr_size/2])
    cov = np.array([[arr_size*2.5,0],[0,arr_size*2.5]])
    rv = multivariate_normal(means,cov)
    response = rv.pdf(pos)

    return response

def get_geom(rover_height:float,rover_reach:float,square_size:float,arr_size:float,solar_angle:float):
    ds = square_size/arr_size

    count_arr = np.dstack(np.meshgrid(np.arange(arr_size),np.arange(arr_size)))
    y_dist = np.abs(ds*count_arr[:,:,0]-(square_size/2))
    x_dist = rover_reach+ds*count_arr[:,:,1]
    total_dist = np.sqrt(x_dist**2+y_dist**2)
    
    e_map = (np.pi/2)-np.arctan(total_dist/rover_height)
    i_map = ((np.pi/180)*solar_angle)*np.ones(e_map.shape)
    return e_map,i_map

def get_timeseries():
    arr_size = 250
    sim_steps = 100
    sim_obj = SimFrame((arr_size,arr_size),0.01,1,sim_steps)
    sim_array = sim_obj.runsim()
    w = 0.66
    e,i = get_geom(1,1,0.01,arr_size,45)

    rfl_array = np.zeros(sim_array.shape)
    for n in range(sim_array.shape[2]):
        hapke_obj = HapkeFrame(sim_array[:,:,n])
        rfl_frame = hapke_obj.hapke_reflectance(w,i,e)
        rfl_array[:,:,n] = rfl_frame
    
    w_range = np.arange(0,1,0.02)
    w_rfl_array = np.zeros((sim_steps,w_range.shape[0]))
    for n in (pbar := tqdm(range(sim_array.shape[2]))):
        pbar.set_description("Getting w plot")
        hapke_obj = HapkeFrame(sim_array[:,:,n])
        for count,j in enumerate(w_range):
            rfl_frame = hapke_obj.hapke_reflectance(j,i,e)
            measured_rfl = np.sum(np.multiply(rfl_frame,get_instrument_response(arr_size)))
            w_rfl_array[n,count] = measured_rfl

    return rfl_array,sim_array,w_range,w_rfl_array

def get_sensitivity():
    arr_size = 250
    sim_steps = 100
    w_range = np.arange(0,1,0.02)
    e,i = get_geom(1,1,0.01,arr_size,45)

    m_carlo_tests = 100
    w_rfl_array = np.zeros((m_carlo_tests,w_range.shape[0]))
    for j in (pbar := tqdm(range(m_carlo_tests))):
        pbar.set_description("Running Monte-Carlo Simulation")
        k=0.01*0.15
        crater_rate = 0.01+np.random.randn()*k
        sim_obj = SimFrame((arr_size,arr_size),0.01,1,sim_steps)
        sim_array = sim_obj.runsim()
        hapke_obj = HapkeFrame(sim_array[:,:,-1])
        for count,w in enumerate(w_range):
            rfl_frame = hapke_obj.hapke_reflectance(w,i,e)
            measured_rfl = np.sum(np.multiply(rfl_frame,get_instrument_response(arr_size)))
            w_rfl_array[j,count] = measured_rfl

    return w_range,w_rfl_array

if __name__ == "__main__":
    arr = get_instrument_response(250)
    plt.imshow(arr)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('instrument_response.png')
    # rfl_array,sim_array,w_x,w_y = get_timeseries()
    # np.save('rfl_array.npy',rfl_array)
    # np.save('sim_array.npy',sim_array)
    # np.save('w_y.npy',w_y)
    # for i in range(10):
    #     plt.plot(w_x,w_y[10*i,:])
    # plt.show()
    # fig,[ax1,ax2,ax3] = plt.subplots(1,3)
    # art_list1 = []
    # art_list2 = []
    # art_list3 = []
    # for i in range(rfl_array.shape[2]):
    #     art1 = ax1.imshow(rfl_array[:,:,i],vmin=rfl_array.min(),vmax=rfl_array.max(),cmap='Reds')
    #     art2 = ax2.imshow(sim_array[:,:,i],vmin=0.25,vmax=0.83,cmap='Reds')
    #     art3 = ax3.plot(w_x,w_y[i,:])
    #     art_list1.append([art1,art2,art3])

    # anim1 = ArtistAnimation(fig,art_list1,interval=50,blit=True,repeat=False)
    # # anim2 = ArtistAnimation(fig,art_list2,interval=50,blit=True,repeat=False)
    # # anim3 = ArtistAnimation(fig,art_list3,interval=50,blit=True,repeat=False)
    # plt.show()



