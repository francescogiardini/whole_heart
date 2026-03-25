import numpy as np
from analyze_R_matrices import compute_threshold_percentages, plot_threshold_percentages

fa_data = {
    ('N1','L'): np.array([0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, np.nan, np.inf]),
    ('N1','R'): np.array([0.02, 0.12, 0.22, 0.32, 0.42]),
    ('N1','D'): np.array([0.01, 0.11, 0.21, 0.31, 0.41]),
    ('N2','L'): np.array([0.2, 0.3, 0.4, 0.5]),
    ('N10','R'): np.array([0, 0, np.nan]),
}

thresholds = [0.1,0.2,0.3,0.4,0.5]
results, df = compute_threshold_percentages(fa_data, thresholds=thresholds, exclude_zeros=True)
print('DataFrame:')
print(df)

fig, axes = plot_threshold_percentages(results,
                                       thresholds=thresholds,
                                       outpath='/home/helios/ubuntu_data/Dual_MesoSPIM/analysis_segmented/R_matrices_segm/test_thresholds_plot.png')
print('Plot saved to test_thresholds_plot.png')

