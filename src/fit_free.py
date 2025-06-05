import numpy as np

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
    decay_curve = decay_curve - offset
    # clip the timebin to above or equal to 0
    decay_curve = np.clip(decay_curve, 0, None)
    w = 2*np.pi*f
    G_IRF = np.dot(np.transpose(shifted_irf) , np.cos(w*time_axis)) / np.sum(shifted_irf)
    S_IRF = np.dot(np.transpose(shifted_irf) , np.sin(w*time_axis)) / np.sum(shifted_irf)
    cos_coeff = np.cos(w*time_axis)
    sin_coeff = np.sin(w*time_axis)
    
    # corrected coefficients
    corrected_cos_coeff = (G_IRF/(G_IRF**2 + S_IRF**2))*cos_coeff + (S_IRF/(G_IRF**2 + S_IRF**2))*sin_coeff
    corrected_sin_coeff = (-S_IRF/(G_IRF**2 + S_IRF**2))*cos_coeff + (G_IRF/(G_IRF**2 + S_IRF**2))*sin_coeff

    decay_curve_sum = np.sum(decay_curve)
    G = np.dot(decay_curve, corrected_cos_coeff) / decay_curve_sum
    S = np.dot(decay_curve, corrected_sin_coeff) / decay_curve_sum

    # get the 2nd harmonic
    G_IRF_2nd = np.dot(np.transpose(shifted_irf) , np.cos(2*w*time_axis)) / np.sum(shifted_irf)
    S_IRF_2nd = np.dot(np.transpose(shifted_irf) , np.sin(2*w*time_axis)) / np.sum(shifted_irf)
    corrected_cos_coeff_2nd = (G_IRF_2nd/(G_IRF_2nd**2 + S_IRF_2nd**2))*cos_coeff + (S_IRF_2nd/(G_IRF_2nd**2 + S_IRF_2nd**2))*sin_coeff
    corrected_sin_coeff_2nd = (-S_IRF_2nd/(G_IRF_2nd**2 + S_IRF_2nd**2))*cos_coeff + (G_IRF_2nd/(G_IRF_2nd**2 + S_IRF_2nd**2))*sin_coeff
    G_2nd = np.dot(decay_curve, corrected_cos_coeff_2nd) / decay_curve_sum
    S_2nd = np.dot(decay_curve, corrected_sin_coeff_2nd) / decay_curve_sum

    phi = np.arctan2(G, S) 
    m = np.sqrt(G**2 + S**2)
    tau_phase = 1/w * np.tan(phi)
    tau_m = 1/w * np.sqrt(1/m**2 - 1)
    return G,S, G_2nd, S_2nd, tau_phase, tau_m

