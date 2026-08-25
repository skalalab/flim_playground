import numpy as np
import pytest
import pandas as pd
from src.fov_extraction import extract_fit_results, extract_spcimage_fit_results

def test_extract_fit_results_tm_iw():
    channel_name = "nadh"
    decay_curves = {"cell_1": np.array([100.0, 50.0, 25.0])}
    
    # 2 components
    results_2ch = {
        "amp1": [80.0],
        "amp2": [20.0],
        "t1": [0.4],
        "t2": [2.5],
        "offset": [5.0],
    }
    
    shifted_irf = np.array([1.0, 0.0, 0.0])
    time_axis = np.array([0.0, 1.0, 2.0])
    
    warning_msg, single_cell_features = extract_fit_results(
        channel_name=channel_name,
        decay_curves=decay_curves,
        results=results_2ch,
        num_components=2,
        shifted_irf=shifted_irf,
        time_axis=time_axis,
        start=0,
        end=3,
        fixed_lifetimes=None
    )
    
    # Calculate expected tm_iw
    # amp1 = 80.0, t1 = 0.4, amp2 = 20.0, t2 = 2.5
    # denom = 80.0 * 0.4 + 20.0 * 2.5 = 32.0 + 50.0 = 82.0
    # numerator = 80.0 * 0.4^2 + 20.0 * 2.5^2 = 80.0 * 0.16 + 20.0 * 6.25 = 12.8 + 125.0 = 137.8
    # tm_iw_ns = 137.8 / 82.0 = 1.6804878...
    # tm_iw_ps = 1680.4878...
    expected_tm_iw = (137.8 / 82.0) * 1000
    
    feature_name = f"Lifetime fit_{channel_name}: tm_iw"
    assert feature_name in single_cell_features["cell_1"]
    assert single_cell_features["cell_1"][feature_name] == pytest.approx(expected_tm_iw)

    # 3 components
    results_3ch = {
        "amp1": [50.0],
        "amp2": [30.0],
        "amp3": [20.0],
        "t1": [0.4],
        "t2": [1.5],
        "t3": [4.0],
        "offset": [5.0],
    }
    
    warning_msg, single_cell_features_3 = extract_fit_results(
        channel_name=channel_name,
        decay_curves=decay_curves,
        results=results_3ch,
        num_components=3,
        shifted_irf=shifted_irf,
        time_axis=time_axis,
        start=0,
        end=3,
        fixed_lifetimes=None
    )
    
    # Calculate expected tm_iw for 3 components
    # amp1 = 50.0, t1 = 0.4, amp2 = 30.0, t2 = 1.5, amp3 = 20.0, t3 = 4.0
    # denom = 50.0 * 0.4 + 30.0 * 1.5 + 20.0 * 4.0 = 20.0 + 45.0 + 80.0 = 145.0
    # numerator = 50.0 * 0.16 + 30.0 * 2.25 + 20.0 * 16.0 = 8.0 + 67.5 + 320.0 = 395.5
    # tm_iw_ns = 395.5 / 145.0 = 2.727586...
    # tm_iw_ps = 2727.586...
    expected_tm_iw_3 = (395.5 / 145.0) * 1000
    assert feature_name in single_cell_features_3["cell_1"]
    assert single_cell_features_3["cell_1"][feature_name] == pytest.approx(expected_tm_iw_3)


def test_extract_spcimage_fit_results_tm_iw(monkeypatch):
    # Mock load_image to return dummy 2D arrays
    # 2 components: a1, t1, t2
    # Background (0) is masked, so we set some values
    t1_img = np.array([[0.4, 0.4], [0.4, 0.4]])
    a1_img = np.array([[80.0, 80.0], [80.0, 80.0]])
    t2_img = np.array([[2.5, 2.5], [2.5, 2.5]])
    mask_img = np.array([[1, 1], [2, 2]])
    
    images = {
        "ch1_Mask": mask_img,
        "ch1_SPCImage t1": t1_img,
        "ch1_a1": a1_img,
        "ch1_t2": t2_img,
    }
    
    monkeypatch.setattr("src.fov_extraction.load_image", lambda path: images[path])
    
    metadata = {
        "ch1_Mask": "ch1_Mask",
        "ch1_SPCImage t1": "ch1_SPCImage t1",
        "ch1_a1": "ch1_a1",
        "ch1_t2": "ch1_t2",
        "fov_col": "fov_1"
    }
    
    err, df = extract_spcimage_fit_results(metadata, "ch1", 2, "fov_col")
    assert err == ""
    
    # expected tm_iw is:
    # alpha1 = 0.8, alpha2 = 0.2
    # denom = 0.8 * 0.4 + 0.2 * 2.5 = 0.32 + 0.50 = 0.82
    # num = 0.8 * 0.16 + 0.2 * 6.25 = 0.128 + 1.25 = 1.378
    # tm_iw = 1.378 / 0.82 = 1.6804878...
    # The output is converted to float
    expected_tm_iw = 1.378 / 0.82
    
    col_name = "Lifetime fit_ch1: tm_iw"
    assert col_name in df.columns
    assert df.loc["fov_1_1", col_name] == pytest.approx(expected_tm_iw)
