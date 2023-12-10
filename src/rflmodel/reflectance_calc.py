'''
Module for calculating the reflectance value of the resulting space weathering frame
'''
import pyhapke
import pyhapke.constants as hapkeconst
import numpy as np


class HapkeFrame:
    def __init__(self,porosity:np.ndarray):
        phi = 1-porosity
        self.K = -np.log(1-1.209*phi**(2/3))/(1.209*phi**(2/3)) #porosity coefficient from Hapke, 2012 Eq. 7.45b

    def H_approx(self,x:np.ndarray,gamma:float):
        '''
        Given x (either mu or mu_0) and the albedo factor, return the approximate Chandrasekhar H function from Hapke, 2012 Eq. 8.70b 
        '''
        return (1+2*x/self.K)/(1+2*gamma*x/self.K)

    def hapke_reflectance(self,w:float,i:np.ndarray,e:np.ndarray):
        '''
        Given the incident angle, i, emission angle, and single scattering albedo, calculate the reflectance of a pixel with a given prosity.
        '''

        gamma = np.sqrt(1-w) #Albedo factor from Hapke, 2012 Eq. 7.21b

        mu0 = np.cos(i) #See pg. 174 of Hapke, 2012
        mu = np.cos(e)

        H = self.H_approx(mu,gamma)
        H0 = self.H_approx(mu0,gamma)

        p_reg = 0.8098 #Given by pyhapke model

        return self.K * (w/(4*np.pi)) * (mu0/(mu0+mu)) * (p_reg + H0*H - 1)
    
if __name__ == "__main__":
    por_arr = 0.3*np.ones((100,100))
    e_arr = (np.pi/4)*np.ones((100,100))
    i_arr = (np.pi/4)*np.ones((100,100))
    w = 0.66

    obj = HapkeFrame(por_arr)
    rfl_im = obj.hapke_reflectance(w,i_arr,e_arr)
    print((rfl_im==rfl_im[0,0]).all())




    

        
