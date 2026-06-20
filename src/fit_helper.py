import numpy as np

def upsample_irf(irf, scale=10):
    return np.interp(np.linspace(0, len(irf), len(irf)*scale), np.arange(len(irf)), irf)

def irf_fwhm_bins(irf):
    """Full Width at Half Maximum of the IRF main peak, in bins.

    Walks left/right from the peak until the value drops below half-max,
    returning the contiguous half-max region's width. This isolates the
    main peak even when after-pulses or shoulders also exceed half-max,
    as long as they're separated from the peak by a sub-half-max gap.

    Used as the natural length scale for the shift-fit halfwidth: shifts
    smaller than the IRF's own width still represent meaningful IRF/decay
    alignment, while larger shifts move the IRF off the data entirely.
    """
    irf = np.asarray(irf, dtype=float)
    peak = float(irf.max())
    if peak <= 0:
        raise ValueError("IRF max is non-positive; cannot compute FWHM.")
    p = int(np.argmax(irf))
    half = peak / 2.0
    L = p
    while L > 0 and irf[L - 1] >= half:
        L -= 1
    R = p
    while R < len(irf) - 1 and irf[R + 1] >= half:
        R += 1
    # returns the 2 times of the FWHM just to be safe
    return max(2, 2*(R - L))

def irf_shift(irf, shift, irf_upsampled=None):
    scale = 10
    if irf_upsampled is None:
        irf_upsampled = upsample_irf(irf, scale)
    # shift the irf curve
    irf_shifted = np.roll(irf_upsampled, int(shift * scale))
    # downsample the irf curve back to original size
    irf_shifted_downsampled = irf_shifted[::scale]

    irf_shifted_downsampled /= np.sum(irf_shifted_downsampled)
   
    return irf_shifted_downsampled

def forward_pass(amp1, t1, offset, shifted_irf, time_axis, amp2=None, t2=None, amp3=None, t3=None):
    #t_i is in ns
    if amp2 is not None and amp3 is not None and t2 is not None and t3 is not None:
        decay = amp1 * np.exp(-time_axis / t1) + amp2 * np.exp(-time_axis / t2) +  amp3 * np.exp(-time_axis / t3) + offset
    elif amp2 is not None and t2 is not None:
        decay = amp1 * np.exp(-time_axis / t1) + amp2 * np.exp(-time_axis / t2) + offset
    else:
        decay = amp1 * np.exp(-time_axis / t1) + offset
    convolved_decay = np.fft.ifft(np.fft.fft(decay) * np.fft.fft(shifted_irf)).real
    
    return convolved_decay
def nll_poisson(fitted, data, start, end):
    if end == -1 or end > len(data):
        end = len(data) 
    fitted = np.maximum(fitted, 1e-12)  # Prevent log(0)
    likelihood = -np.sum(data[start:end] * np.log(fitted[start:end]) - fitted[start:end])
    return likelihood

def reduced_chi_square(fitted, data, start, end, num_free_params):
    if end == -1 or end > len(data):
        end = len(data) 
    # crop out the region of interest by time gates
    data_slice = data[start:end]
    fitted_slice = fitted[start:end]
    # keep only bins where fitted > 0 to avoid division by zero
    valid = (fitted_slice > 0)
    data_slice = data_slice[valid] 
    fitted_slice = fitted_slice[valid]
    residuals = data_slice - fitted_slice
    tmp_chiq = residuals**2/fitted_slice
    chiq = tmp_chiq.sum() / (len(data_slice) - num_free_params) 
    return chiq

def objective(params, data, irf, time_axis, start=0, end=-1, fitting_algo="MLE", irf_upsampled=None):
    if 'shift' in params:
        shift = params['shift']
        irf = irf_shift(irf, shift, irf_upsampled=irf_upsampled)
    # otherwise, the irf should be already shifted
    if end == -1 or end > len(data):
        end = len(data)
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
        return nll_poisson(fitted, data, start, end)
    elif fitting_algo == "WLS":
        # Pearson weighting: variance estimated from the model (fitted), so the
        # minimized objective matches the reported reduced χ² (also Pearson).
        weights = np.sqrt(np.maximum(fitted[start:end], 1))
        residuals = (data[start:end] - fitted[start:end]) / weights
        return residuals


def create_progress_callback(progress_bar):
    def progress_callback(current, total):
        progress = (current + 1) / total
        progress_bar.progress(progress)
    return progress_callback