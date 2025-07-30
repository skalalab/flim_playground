import numpy as np
from phasorpy import phasor

def get_phasor_features(decay_curve, shifted_irf, time_axis, f=0.08, offset=0):

    """
    Calculate the phasor features for a given decay curve
    Args:
        decay_curve: the decay curve to be fitted
        shifted_irf: the shifted irf
        time_axis: the time axis
        f: laser repetition rate in [GHz]
        offset: the offset of the decay curve
        harmonic: the harmonic of the decay curve
    """
    # step 1: subtract the esetimated offset and clip the timebin to above or equal to 0
    decay_curve = decay_curve - offset
    # clip the timebin to above or equal to 0
    decay_curve = np.clip(decay_curve, 0, None)

    # step 2: calculate the raw phasor coordinates
    _, g_raw, s_raw = phasor.phasor_from_signal(decay_curve)
    _, g_raw_2nd, s_raw_2nd = phasor.phasor_from_signal(decay_curve, harmonic=2)
    # step 3: calculate the phasor of irf
    _, g_irf, s_irf = phasor.phasor_from_signal(shifted_irf)
    _, g_irf_2nd, s_irf_2nd = phasor.phasor_from_signal(shifted_irf, harmonic=2)
    # step 4: use phasor.divide to correct the phasor coordinates
    G, S = phasor.phasor_divide(g_raw, s_raw, g_irf, s_irf)
    G_2nd, S_2nd = phasor.phasor_divide(g_raw_2nd, s_raw_2nd, g_irf_2nd, s_irf_2nd)
    w = 2*np.pi*f
    phi = np.arctan2(G, S) 
    m = np.sqrt(G**2 + S**2)
    tau_phase = 1/w * np.tan(phi)
    tau_m = 1/w * np.sqrt(1/m**2 - 1)
    return G,S, G_2nd, S_2nd, tau_phase, tau_m



#   # step 2: calculate the phasor
#     G_IRF = np.dot(np.transpose(shifted_irf) , np.cos(w*time_axis)) / np.sum(shifted_irf)
#     S_IRF = np.dot(np.transpose(shifted_irf) , np.sin(w*time_axis)) / np.sum(shifted_irf)
#     print(f"my: G_IRF: {G_IRF}, S_IRF: {S_IRF}, np.sqrt(G_IRF**2 + S_IRF**2): {np.sqrt(G_IRF**2 + S_IRF**2)}")
#     print(f"phasorpy: g_irf: {g_irf}, s_irf: {s_irf}, np.sqrt(g_irf**2 + s_irf**2): {np.sqrt(g_irf**2 + s_irf**2)}")
#     cos_coeff = np.cos(w*time_axis)
#     sin_coeff = np.sin(w*time_axis)
    
#     # corrected coefficients
#     corrected_cos_coeff = (G_IRF/(G_IRF**2 + S_IRF**2))*cos_coeff + (S_IRF/(G_IRF**2 + S_IRF**2))*sin_coeff
#     corrected_sin_coeff = (-S_IRF/(G_IRF**2 + S_IRF**2))*cos_coeff + (G_IRF/(G_IRF**2 + S_IRF**2))*sin_coeff

#     decay_curve_sum = np.sum(decay_curve)
#     G = np.dot(decay_curve, corrected_cos_coeff) / decay_curve_sum
#     S = np.dot(decay_curve, corrected_sin_coeff) / decay_curve_sum

#     # get the 2nd harmonic
#     G_IRF_2nd = np.dot(np.transpose(shifted_irf) , np.cos(2*w*time_axis)) / np.sum(shifted_irf)
#     S_IRF_2nd = np.dot(np.transpose(shifted_irf) , np.sin(2*w*time_axis)) / np.sum(shifted_irf)
#     corrected_cos_coeff_2nd = (G_IRF_2nd/(G_IRF_2nd**2 + S_IRF_2nd**2))*cos_coeff + (S_IRF_2nd/(G_IRF_2nd**2 + S_IRF_2nd**2))*sin_coeff
#     corrected_sin_coeff_2nd = (-S_IRF_2nd/(G_IRF_2nd**2 + S_IRF_2nd**2))*cos_coeff + (G_IRF_2nd/(G_IRF_2nd**2 + S_IRF_2nd**2))*sin_coeff
#     G_2nd = np.dot(decay_curve, corrected_cos_coeff_2nd) / decay_curve_sum
#     S_2nd = np.dot(decay_curve, corrected_sin_coeff_2nd) / decay_curve_sum