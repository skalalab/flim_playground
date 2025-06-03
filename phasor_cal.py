import tifffile as tiff
import numpy as np
from src.sdt_io import read_sdt150
import pandas as pd
from src.fit_helper import irf_shift, forward_pass
from src.fit import fit_curves
from src.phasor import get_gs_coords

def plot_fit(result, time_axis, cell_summed_decay, shifted_irf):
    import matplotlib.pyplot as plt
    plt.plot(cell_summed_decay)
    amp1 = result["amp1"][0]
    t1 = result["t1"][0]
    offset = result["offset"][0]
    if "amp2" in result:
        amp2 = result["amp2"][0]
        t2 = result["t2"][0]
    plt.plot(forward_pass(amp1, t1, offset, shifted_irf, time_axis, amp2, t2))
    plt.show()


rt = r"\\skala-dv1.discovery.wisc.edu\ws\skala\0-Projects and Experiments\RD - Irg-Mydgf\Phasor analysis - zebrafish-selected\Mydgf N1\mydgf N1_sdt and masks"
rt = r"\\skala-dv1.discovery.wisc.edu\ws\skala\0-Projects and Experiments\RD - Irg-Mydgf\Phasor analysis - zebrafish-selected\Mydgf N2\mydgf N2_sdt and masks"
rt = r"\\skala-dv1.discovery.wisc.edu\ws\skala\0-Projects and Experiments\RD - Irg-Mydgf\Phasor analysis - zebrafish-selected\Mydgf N3\mydgf N3_sdt and masks"
rt = r"\\skala-dv1.discovery.wisc.edu\ws\skala\0-Projects and Experiments\RD - Irg-Mydgf\Phasor analysis - zebrafish-selected\Irg1 N2\Irg1 N2_sdt and masks"
rt = r"\\skala-dv1.discovery.wisc.edu\ws\skala\0-Projects and Experiments\RD - Irg-Mydgf\Phasor analysis - zebrafish-selected\Irg1 N3\Irg1 N3_sdt and masks"
import os
image_metadata = pd.read_csv(os.path.join(rt, "image_metadata.csv"))
duration = 10
time_bins = 256
cell_phasor = {}
for i, row in image_metadata.iterrows():
    image_name = row["image_name"]
    mask_path = row["mask"]
    mask = tiff.imread(mask_path)
    decay_path = row["nadh decay"]
    decay = read_sdt150(decay_path)
    irf_path = row["nadh irf"]
    irf = np.loadtxt(irf_path)
    shift = row["shift"]
    shifted_irf = irf_shift(irf, shift)
    cell_labels = np.unique(mask)
    cell_labels = cell_labels[cell_labels != 0]
    count = 0
    for cell_label in cell_labels:
        cell_mask = mask == cell_label
        cell_summed_decay = decay[cell_mask, :].sum(axis=0)
        # get the offset by fitting 
        result = fit_curves(duration, time_bins, [cell_summed_decay], shifted_irf, 2, "WLS", fit_shift=False, start=40, end=240)
        period = duration / time_bins
        time_axis = np.linspace(0, (time_bins - 1) * period, time_bins, dtype=np.float64)
        count += 1
        if count % 10 == 0:
            plot_fit(result, time_axis, cell_summed_decay, shifted_irf)
        offset = result["offset"][0]
        cell_id = image_name + "_" + str(cell_label)
        cell_phasor[cell_id] = {"offset": offset}
        g, s = get_gs_coords(cell_summed_decay, shifted_irf, f=0.08, duration=10, offset=offset)
        cell_phasor[cell_id]["g"] = g
        cell_phasor[cell_id]["s"] = s
        # calculate tau_phase and tau_m
        w = 2*np.pi*0.08
        phi = np.arctan2(s, g) 
        m = np.sqrt(g**2 + s**2)
        tau_phase = 1/w * np.tan(phi)
        tau_m = 1/w * np.sqrt(1/m**2 - 1)
        cell_phasor[cell_id]["tau_phase"] = tau_phase
        cell_phasor[cell_id]["tau_m"] = tau_m
    

# write the dictionary as a csv file, use key as
df = pd.DataFrame.from_dict(cell_phasor, orient="index")
df.to_csv(os.path.join(rt, "cells_phasor.csv"))





