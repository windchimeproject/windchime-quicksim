'''Windchime Simulation Module'''
#Just imported these for future use
import json
from numba import njit
import numpy as np
from scipy import signal
from scipy import stats
from tqdm import tqdm, trange


'''
Here we initialize some important constants that are used as defaults
'''
G = 6.64730e-11 #m^3/kg/s^2
amu = 1.6605390666e-29 #kg
k_B = 1.380649e-23 #J/K
GeV_per_c2 = 1.7826619218835431593e-27
mass_dm = 2.176434e-8 #kg https://en.wikipedia.org/wiki/Planck_units

sensor_mass = 1e-3 #kg
sensor_density = 1e4 #kg/m^3
min_impact_parameter = (sensor_mass/sensor_density)**(1/3) #metres
gas_pressure = 1e-10 #Pa
A_d = (sensor_mass/sensor_density)**(2/3) #sensor cross sectional area
beta = gas_pressure*A_d*np.sqrt(4*amu*k_B*4)/sensor_mass**2

radius = 5 #metre
v_dm = 220e3 #m/s
rho_dm = 0.3*GeV_per_c2*(100**3) #kg/m^3
expected_rate_through_radius = rho_dm*v_dm/mass_dm*3600*24*365*np.pi*radius**2 #events per year

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



def signal(b, v, sensor_vectors, b_bases, v_bases, mass, min_impact_parameter, G=6.67e-11):
    '''the S of the SNR. Based on
    https://github.com/windchimeproject/documentation_and_notes/blob/main/analysis_notes/Analytic_SNR_for_a_single_sensor.pdf
    
    Parameters:
    b: array of impact parameters.
    v: array of velocity magnitudes.
    sensor_vectors: sensor orientations. 3xM array; individual vectors assumed to be normalised.
    b_bases: b basis vectors for each entry, as defined in find_basis_vectors.
             3xM array; individual vectors assumed to be normalised.
    v_bases: v basis vectors for each entry, as defined in find_basis_vectors.
             3xM array; individual vectors assumed to be normalised.
    mass: dark matter particle mass.
    min_impact_parameter: Smallest allowable impact factor. Smaller impact factors are set to this value.
    '''
    b_dot_n = np.einsum('ij,ij->j',sensor_vectors,b_bases) #dot product
    v_dot_n = np.einsum('ij,ij->j',sensor_vectors,v_bases)
    b[b < min_impact_parameter] = min_impact_parameter
    return G**2*mass**2*np.pi*(3*b_dot_n**2 + v_dot_n**2)/(8*b**3*v)

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

def beta_func(gas_pressure=gas_pressure, A_d=A_d, sensor_mass=sensor_mass, T=4):
    '''
    Noise model from https://arxiv.org/abs/1903.00492, for free-falling sensors.
    gas_pressure: the gas pressure
    A_d: sensor cross-sectional area
    sensor_mass: mass of sensors
    T: temperature in Kelvin
    '''
    return gas_pressure*A_d*np.sqrt(4*amu*k_B*T)/sensor_mass**2
    
def simulate(track_parameters, sensor_positions, 
             beta,
             bins_snr,
             bins_b,
             return_all_snrs = False):
    '''
    track_parameters: stack of the track parameters from previous func
    sensor_positions: positions of the sensors
    beta: noise parameter defined by beta_func
    bins_snr: bins for storing the SNR histograms
    bins_b: bins for storing impact parameter histograms
    return_all_snrs: set False if want full detector SNR, set True for individual SNRs
    '''
    # Tracks is (7, N) for N tracks where dimensions are velocity, entry 3-position, exit 3-position
    
    assert track_parameters.shape[0] == 7, "Track format is (velocity, 3-entry, 3-exit) for 7xNtrack"
    
    # First dimension is signal, second is impact parameter, third is noise
    N_sensors = sensor_positions.shape[1]
    N_vels = len(track_parameters[0, :])
    snr_bin_data = np.zeros((bins_snr.shape))
    b_bin_data = np.zeros((bins_b.size))
    sqrt_noise_bin_data = np.zeros((bins_snr.size))
    
    if return_all_snrs == True:
        snr_all_data = np.zeros((N_sensors, N_vels))
        b_all_data = np.zeros((N_sensors, N_vels))
    
    for i, track_parameter in tqdm(enumerate(track_parameters.T)): 
        velocity = track_parameter[0]
        position_entry = track_parameter[1:4]
        position_exit = track_parameter[4:]
        
        #print(i, velocity, position_entry, position_exit)
        assert sensor_positions.shape[0] == 3
        # Call some function to get SNRs
        b_bases, v_bases, b_vecs = find_basis_vectors(sensor_positions, np.repeat([position_entry], N_sensors, axis=0).T, np.repeat([position_exit], N_sensors, axis=0).T)
        b = np.linalg.norm(b_vecs,axis=0)
        signal_result = signal(b, velocity, sensor_positions, b_bases, v_bases, mass=mass_dm, min_impact_parameter=min_impact_parameter)
        
        snr_bin_data += np.histogram(np.sum(signal_result, axis=0),
                                           bins=bins_snr)[1]
        b_bin_data += np.histogram(np.min(b), 
                                           bins=bins_b)[1]
        sqrt_noise_bin_data += np.histogram(np.sum(signal_result, axis=0)*beta,
                                           bins=bins_snr)[1]
        
        if return_all_snrs == True:
            snr_all_data[:,i] = signal_result
            b_all_data[:,i] = b
        
        # stop early while testing
#         if i > 100: break
        
        snr_bin_data /= track_parameters.shape[1] # Normalize by the number of tracks
        b_bin_data /= track_parameters.shape[1]
        sqrt_noise_bin_data /= track_parameters.shape[1]
    if return_all_snrs == True:
        return snr_all_data, b_all_data
    else:
        return snr_bin_data/np.sqrt(sqrt_noise_bin_data), b_bin_data, sqrt_noise_bin_data

    
@njit
def SNRs_from_S_and_beta(S, beta):
    '''Calculates the SNR from the output of array_SNRs, if return_individual_SNR=True. Vars defined in previus funcs'''
    S_summed = np.sum(S, axis=0)
    return np.sqrt(S_summed/beta)

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


def dm_events_per_year(mass_dm=mass_dm, rho_dm=rho_dm, v_dm=v_dm):
    '''
    Caculates expected DM events per year
    '''
    expected_rate_through_radius = rho_dm*v_dm/mass_dm*3600*24*365*np.pi*radius**2
    return expected_rate_through_radius
