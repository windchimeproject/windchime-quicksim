'''
Windchime Simulation Module

This module sends a large amount of tracks through a given accelerometer array and returns impact parameters and SNR. 
This is used to determine thermal sensitivity conditions.
'''
#Imports, some for future use
import json
import warnings
from numba import njit
import numpy as np
from scipy import stats
from tqdm import tqdm, trange
from scipy import constants


'''
Here we initialize some important constants that are used as defaults
'''
G = 6.64730e-11 #m^3/kg/s^2
amu = 1.6605390666e-27 #kg
k_B = 1.380649e-23 #J/K
GeV_per_c2 = 1.7826619218835431593e-27

v_dm = 220e3 #m/s
rho_dm = 0.3*GeV_per_c2*(100**3) #kg/m^3

def find_basis_vectors(sensor_pos, entry_vecs, exit_vecs, #only_one_track=False
                      ):
    '''
    Find basis vectors and impact vectors.
    The v basis is simply the normalised velocity vector.
    The b basis is the normalised b vector, and the b vector is the vector that points from the sensor to the
    DM particle track, and is orthogonal to said track.
    
    Parameters:
    sensor_pos: coordinates of sensors, 3xN array for N entries.
    entry_vecs: vectors indicating the point at which tracks enter the analysis sphere,
                3xM array for M entries
    exit_vecs: vectors indicating the point at which tracks exit the analysis sphere,
               3xM array for M entries
    '''
    v_dirs = exit_vecs - entry_vecs
    v_bases = v_dirs/np.linalg.norm(v_dirs, axis=0)
    r_disp = entry_vecs-sensor_pos
    d_vecs = np.einsum('ij,ij->j',r_disp,v_bases)*v_bases #dot product
    b_vecs = r_disp - d_vecs
    b_bases = b_vecs/np.linalg.norm(b_vecs, axis=0)

    return (b_bases, v_bases, b_vecs)



def signal(b, v, sensor_vectors, b_bases, v_bases, mass, min_impact_parameter):
    '''the S of the SNR. Based on
    https://github.com/windchimeproject/documentation_and_notes/blob/main/analysis_notes/Analytic_SNR_for_a_single_sensor.pdf
    
    Parameters:
    b: array of impact parameters.
    v: array of velocity magnitudes.
    sensor_vectors: sensor orientations. 3xN array; individual vectors assumed to be normalised.
    b_bases: b basis vectors for each entry, as defined in find_basis_vectors.
             3xN array; individual vectors assumed to be normalised.
    v_bases: v basis vectors for each entry, as defined in find_basis_vectors.
             3xN array; individual vectors assumed to be normalised.
    mass: dark matter particle mass.
    min_impact_parameter: Smallest allowable impact factor. Smaller impact factors are set to this value.
    '''
    b_dot_n = np.einsum('ij,ij->j',sensor_vectors,b_bases) #dot product
    v_dot_n = np.einsum('ij,ij->j',sensor_vectors,v_bases)
#     b[b < min_impact_parameter] = min_impact_parameter
#     return G**2*mass**2*np.pi*(3*b_dot_n**2 + v_dot_n**2)/(8*b**3*v)
    mass_arr = np.array([mass]*b_bases.shape[1])
    mass_arr[b < min_impact_parameter] = mass_arr[b < min_impact_parameter]/(min_impact_parameter)**3*b[b < min_impact_parameter]**3 #scaling DM mass is equivalent to scaling test mass.
    return G**2*mass_arr**2*np.pi*(3*b_dot_n**2 + v_dot_n**2)/(8*b**3*v)

def signal_without_template_matching(b, v, sensor_vectors, b_bases, v_bases, mass, min_impact_parameter):
    '''Simple signal without template matching.
    
    Parameters:
    b: array of impact parameters.
    v: array of velocity magnitudes.
    sensor_vectors: sensor orientations. 3xN array; individual vectors assumed to be normalised.
    b_bases: b basis vectors for each entry, as defined in find_basis_vectors.
             3xN array; individual vectors assumed to be normalised.
    v_bases: v basis vectors for each entry, as defined in find_basis_vectors.
             3xN array; individual vectors assumed to be normalised.
    mass: dark matter particle mass.
    min_impact_parameter: Smallest allowable impact factor. Smaller impact factors are set to this value.
    '''
    b_dot_n = np.einsum('ij,ij->j',sensor_vectors,b_bases) #dot product
#    v_dot_n = np.einsum('ij,ij->j',sensor_vectors,v_bases)
#     b[b < min_impact_parameter] = min_impact_parameter
#     return G**2*mass**2*np.pi*(3*b_dot_n**2 + v_dot_n**2)/(8*b**3*v)
    mass_arr = np.array([mass]*b_bases.shape[1])
    mass_arr[b < min_impact_parameter] = mass_arr[b < min_impact_parameter]/(min_impact_parameter)**3*b[b < min_impact_parameter]**3 #scaling DM mass is equivalent to scaling test mass.
    return np.sqrt(2)*G*mass_arr*b_dot_n/(b*v) #sqrt 2 comes from integrating signal from t=-b/v to t=b/v


def track_parameter_stacker(vel, entry_vecs, exit_vecs):
    '''
    Just makes a stack to have track parameters in one place
    
    vel: velocities list
    entry_vecs: entry vectors list
    exit_vecs: exit_vectors list
    Outputs 7xN stack (N number of tracks)
    '''
    velocity = vel
    position_entry = entry_vecs
    position_exit = exit_vecs
    track_parameters = np.vstack((velocity, position_entry, position_exit))
    return track_parameters

def beta_func(gas_pressure, A_d, sensor_mass, T=4):
    '''
    Noise model from https://arxiv.org/abs/1903.00492, for free-falling sensors.
    
    gas_pressure: the gas pressure
    A_d: sensor cross-sectional area
    sensor_mass: mass of sensors
    T: temperature in Kelvin
    '''
    return gas_pressure*A_d*np.sqrt(4*amu*k_B*T)/sensor_mass**2

def beta_func_with_mechanical_noise(gas_pressure, A_d, sensor_mass, Q, f_m, T=4):
    '''Noise model from https://arxiv.org/abs/1903.00492, for free-falling sensors.'''
    return (gas_pressure*A_d*np.sqrt(4*amu*k_B*T) + 4*sensor_mass*k_B*T*(f_m*2*np.pi/Q))/sensor_mass**2
    
def simulate(track_parameters, sensor_vectors, sensor_positions,
             sensor_mass, min_impact_parameter,
             beta,
             bins_snr,
             bins_b, mass_dm,
             all_sensor_values = True, signal=signal, check_assertions=True):
    '''
    This runs tracks given by track_parameters through the array and returns the impact parameters and sensor vectors.
    
    track_parameters: stack of the track parameters from previous func
    sensor_vector: orientations of the sensors
    beta: noise parameter defined by beta_func
    bins_snr: bins for storing the SNR histograms
    bins_b: bins for storing impact parameter histograms
    all_sensor_values: set True for SNR histogram of all sensors, False for sum SNR and minimum impact parameter histogram
    '''
    # Tracks is (7, N) for N tracks where dimensions are velocity, entry 3-position, exit 3-position
    
    assert track_parameters.shape[0] == 7, "Track format is (velocity, 3-entry, 3-exit) for 7xNtrack"
    
    # First dimension is signal, second is impact parameter, third is noise
    N_sensors = sensor_positions.shape[1]
    N_vels = len(track_parameters[0, :])
    snr_bin_data = np.zeros((bins_snr.size-1))
    b_bin_data = np.zeros((bins_b.size-1))
    
    if all_sensor_values:
        signal_arr = np.zeros((N_sensors, N_vels))
        b_arr = np.zeros((N_sensors, N_vels))
    
    for i, track_parameter in enumerate(tqdm(track_parameters.T)): 
        velocity = track_parameter[0]
        position_entry = track_parameter[1:4]
        position_exit = track_parameter[4:]
        
        #print(i, velocity, position_entry, position_exit)
        assert sensor_positions.shape[0] == 3
        # Call some function to get SNRs
        b_bases, v_bases, b_vecs = find_basis_vectors(sensor_positions, np.repeat([position_entry], N_sensors, axis=0).T, np.repeat([position_exit], N_sensors, axis=0).T)
        b = np.linalg.norm(b_vecs,axis=0)
        signal_result = signal(b, velocity, sensor_vectors, b_bases, v_bases, mass=mass_dm, min_impact_parameter=min_impact_parameter)
        
        if not all_sensor_values:
            signal_result = np.sum(signal_result, axis=0)
            b = np.min(b)
            
        # Populate histogram with one (if one) or more values (if all sensors)
        snr_result = np.sqrt(signal_result/beta)
        
        if check_assertions:
            # Some sanity checks to check results in bin bounds
            assert bins_snr[0] < snr_result.min(), snr_result.min()
            assert bins_snr[-1] > snr_result.max(), snr_result.max()
            assert bins_b[0] < b.min(), b.min()
            assert bins_b[-1] > b.max(), b.max()    
        
        # Make function just returning histogram values, then use that output in the += line.  THis function called by for loop.
        
        # TODO: Instead of adding here, make list of histogram values.... so .append(np.histogram(snr_result.....))
        snr_bin_data += np.histogram(snr_result, bins=bins_snr)[0]
        b_bin_data += np.histogram(b, bins=bins_b)[0]
        
        if all_sensor_values:
            signal_arr[:,i] = signal_result
            b_arr[:,i] = b
        
    snr_bin_data /= track_parameters.shape[1] # Normalize by the number of tracks
    b_bin_data /= track_parameters.shape[1]
    
    if not all_sensor_values:
        return snr_bin_data, b_bin_data
    if all_sensor_values:
        return snr_bin_data, b_bin_data, signal_arr, b_arr
        
    
@njit
def SNRs_from_S_and_beta(S, beta):
    '''
    Calculates the SNR from the output of array_SNRs, if return_individual_SNR=True. Vars defined in previus funcs
    '''
    S_summed = np.sum(S, axis=0)
    return np.sqrt(S_summed/beta)

def impulse_noise(b, sensor_mass, gas_pressure, A_d, Q, f_m, T=4, QNR=1):
    '''Get impulse noise array from array of b'''
    alpha = 4*k_B*sensor_mass*T*(f_m*2*np.pi/Q) + gas_pressure*A_d*np.sqrt(4*amu*k_B*T)
    tau_min = 2*b/v_dm
    tau = (np.zeros_like(b)+1)*(2*np.sqrt(constants.hbar*sensor_mass/alpha)/QNR)
    tau_below_min_bool = tau < tau_min
    tau[tau_below_min_bool] = tau_min[tau_below_min_bool]
    if np.any(tau>1/f_m):
        warnings.warn('Result may be invalid as integration time exceeds resonance frequency')
        print('Result may be invalid as integration time exceeds resonance frequency')
    return np.sqrt(4*(constants.hbar*sensor_mass/tau)/QNR**2 + alpha*tau)

def impulse_noise_classical(b, sensor_mass, gas_pressure, A_d, Q, f_m, T=4, S_xx=1e-18):
    '''Get impulse noise array from array of b. S_xx in SI units of m^2/Hz'''
    alpha = 4*k_B*sensor_mass*T*(f_m*2*np.pi/Q) + gas_pressure*A_d*np.sqrt(4*amu*k_B*T)
    tau_min = 2*b/v_dm
    tau = (np.zeros_like(b)+1)*np.sqrt(sensor_mass*(f_m*2*np.pi)**2*(S_xx)/(alpha))
    tau_below_min_bool = tau < tau_min
    tau[tau_below_min_bool] = tau_min[tau_below_min_bool]
    return np.sqrt(sensor_mass**2*(f_m*2*np.pi)**2*S_xx/tau + alpha*tau)


@njit
def SNRs_from_S_and_impulse_noise(S, impulse_noise, sensor_mass):
    '''
    Calculates the SNR from the output of array_SNRs, if return_individual_SNR=True. Vars defined in previus funcs
    '''
    return np.sqrt(np.sum((S*sensor_mass/impulse_noise)**2, axis=0))

def toy_MC_poisson_detection(DM_rate, detection_probability_params, trials=100):
    '''
    Toy MC that returns the number of dark matter particles detected in 1 year of exposure.
    DM_rate: Expected number of events per year
    detection_probability_params: requirements for a positive detection
    trials: number of trials
    '''
    p_variable = stats.beta(*detection_probability_params)
    ps = p_variable.rvs(trials)
    DM_particles = stats.poisson(DM_rate).rvs(trials)
    detected_particles = stats.binom.rvs(DM_particles, ps)
    return detected_particles


def dm_events_per_year(mass_dm, rho_dm, v_dm, radius):
    '''
    Caculates expected DM events per year. Constants given above
    '''
    expected_rate_through_radius = rho_dm*v_dm/mass_dm*3600*24*365*np.pi*radius**2
    return expected_rate_through_radius