from erfa import ufunc
import numpy as np
import astropy.constants as ac
import astropy.units as u
from scipy.optimize import minimize, dual_annealing
from pathlib import Path
import bilby
from astropy.time import Time
from astropy.coordinates import EarthLocation, CartesianRepresentation, SphericalRepresentation

SPEED_OF_LIGHT = (ac.c).si.value

def spherical_to_cartesian(ra, dec):
    return ufunc.s2c(ra, dec)

class TimingUncertainty:
    
    def __init__(self, time, ra, dec):
        
        self.time = time
        self.ra = ra
        self.dec = dec
        self.scaling = 1e6
    
        self.propagation_vector = -spherical_to_cartesian(ra, dec)
    
    def uncertainty(self, new_center_scaled):
        position_delay = np.dot(self.propagation_vector, new_center_scaled * self.scaling) / SPEED_OF_LIGHT

        return np.std(self.time + position_delay)*1e3

    def find_minimum(self, x0):
        res = minimize(
            self.uncertainty, 
            x0 / self.scaling, 
            method = 'Nelder-Mead', 
            tol=1e-6,)
        
        print(f'Uncertainty at x0: {self.uncertainty(x0/self.scaling)*1e3}us')
        print(f'Uncertainty at minimum: {self.uncertainty(res.x)*1e3}us')
        
        return res.x * self.scaling

    def find_global_minimum(self, x0):
        res = dual_annealing(
            self.uncertainty, 
            bounds=[
                (-10, 10),
                (-10, 10),
                (-10, 10),
            ], 
            x0=x0/self.scaling,
            maxiter=5000,)
        
        print(f'Uncertainty at x0: {self.uncertainty(x0/self.scaling)*1e3}us')
        print(f'Uncertainty at minimum: {self.uncertainty(res.x)*1e3}us')
        
        return res.x * self.scaling



if __name__ == '__main__':
    
    data_path = Path(__file__).parent.parent / 'data'
    
    result = bilby.core.result.Result.from_hdf5(data_path / 'bilby-NRSur7dq4_prod_data0_1420878141-222656_analysis_H1L1_merge_result.hdf5')
    
    tu = TimingUncertainty(
        result.posterior['geocent_time'],
        result.posterior['ra'],
        result.posterior['dec'],
    )
    
    min1 = CartesianRepresentation(tu.find_global_minimum(np.array([0, 0, 0])) * u.m)
        
    t = Time('2025-01-14 08:22:03')
    x_llo, _ = EarthLocation.of_site('llo').get_gcrs_posvel(t)
    x_lho, _ = EarthLocation.of_site('lho').get_gcrs_posvel(t)
    

    min2 = CartesianRepresentation(tu.find_global_minimum(x_llo.xyz.value)*u.m)
    
    print((x_lho-min1).norm().to(u.km))
    print((x_llo-min1).norm().to(u.km))

    print((x_lho-min2).norm().to(u.km))        
    print((x_llo-min2).norm().to(u.km))        
    
    # x_b = CartesianRepresentation(x_best*u.m)
    
    # pycbc_snrs = {
    #     'H1': 51.265347,
    #     'L1': 57.312855,
    # }
    # for x_m in [
    #         x_llo,
    #         x_lho,
    #         (x_llo * pycbc_snrs['L1'] + x_lho * pycbc_snrs['H1']) / (pycbc_snrs['L1']+pycbc_snrs['H1']),
    #         (x_llo  + x_lho ) /2,        
    #     ]:
    
    
        # print((SphericalRepresentation.from_cartesian(x_b).lat / u.deg).to(u.dimensionless_unscaled))
        # print((SphericalRepresentation.from_cartesian(x_b).lon / u.deg).to(u.dimensionless_unscaled))
        # print((SphericalRepresentation.from_cartesian(x_m).lat / u.deg).to(u.dimensionless_unscaled))
        # print((SphericalRepresentation.from_cartesian(x_m).lon / u.deg).to(u.dimensionless_unscaled))

        # difference = x_b - x_m
        
            
        # print(f'Distance: {difference.norm().to(u.km)}')
        # x_b_unit = x_b / x_b.norm()
        
        # print(difference.dot(x_b_unit))
