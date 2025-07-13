import numpy as np
from scipy.signal import convolve
def irf_shift(irf, shift):
    scale = 10
    irf_upsampled = np.interp(np.linspace(0, len(irf), len(irf)*scale), np.arange(len(irf)), irf)
    # shift the irf curve
    irf_shifted = np.roll(irf_upsampled, int(shift * scale))
    # downsample the irf curve back to original size
    irf_shifted_downsampled = irf_shifted[::scale]

    irf_shifted_downsampled /= np.sum(irf_shifted_downsampled)
   
    return irf_shifted_downsampled

def forward_pass(amp1, t1, offset, shifted_irf, time_axis, amp2=None, t2=None, amp3=None, t3=None):
    # Create the forward model
    # t1 /= 1000  # Convert to ms
    # t2 /= 1000  # Convert to ms
    if amp2 is not None and amp3 is not None and t2 is not None and t3 is not None:
        decay = amp1 * np.exp(-time_axis / t1) + amp2 * np.exp(-time_axis / t2) +  amp3 * np.exp(-time_axis / t3)
    elif amp2 is not None and t2 is not None:
        decay = amp1 * np.exp(-time_axis / t1) +  amp2 *  np.exp(-time_axis / t2)    
    else:
        decay = amp1 * np.exp(-time_axis / t1)
    # convolve the decay with the shifted IRF, finally add the offset
    convolved_decay = convolve(decay, shifted_irf)[:len(time_axis)] + offset
    
    return convolved_decay
def mle_likelihood(fitted, data, start, end):
    fitted = np.maximum(fitted, 1e-10)  # Prevent log(0)
    likelihood = -np.sum(data[start:end] * np.log(fitted[start:end]) - fitted[start:end])
    return likelihood

def chi_square(fitted, data, start=0, end=-1):
    if end == -1 or end > len(data):
        end = len(data) 
    residuals = data - fitted
    residuals[:start] = 0
    residuals[end:] = 0
    chi2 = np.sum((residuals / fitted[start:end])**2)
    non_zero_indices = data > 0
    
    # Use data values as denominator for chi-square calculation
    denominator = data[non_zero_indices]
    residuals = residuals[non_zero_indices]
     # Poisson noise assumption where variance equals the count
    tmp_chiq = residuals**2/denominator
    chiq = tmp_chiq.sum() / len(non_zero_indices) 
    return chiq

def objective(params, data, irf, time_axis, start=0, end=-1, fitting_algo="MLE"):
    if 'shift' in params:
        shift = params['shift']
        irf = irf_shift(irf, shift)
    # otherwise, the irf should be already shifted
    if end == -1 or end > len(data):
        end = len(data) - 1
    amp1 = params['amp1']
    t1 = params['t1']
    offset = params['offset']
    amp2 = params['amp2'] if 'amp2' in params else None
    t2 = params['t2'] if 't2' in params else None
    amp3 = params['amp3'] if 'amp3' in params else None
    t3 = params['t3'] if 't3' in params else None
    # 
    fitted = forward_pass(amp1=amp1, t1=t1, offset=offset, shifted_irf=irf, time_axis=time_axis, amp2=amp2, t2=t2, amp3=amp3, t3=t3)
    # Poisson likelihood
    if fitting_algo == "MLE": 
        return mle_likelihood(fitted, data, start, end)
    elif fitting_algo == "WLS":
        residuals = data[start:end] - fitted[start:end]
        return residuals


def create_progress_callback(progress_bar):
    def progress_callback(current, total):
        progress = (current + 1) / total
        progress_bar.progress(progress)
    return progress_callback