import numpy as np
from analyze_R_matrices import compute_threshold_percentages, plot_rejected_threshold_percentages

fa_data = {
    ('N1','L'): np.array([0.12, 0.22, 0.32, 0.01, 0.5, np.nan]),
    ('N1','R'): np.array([0.05, 0.15, 0.25, 0.35]),
    ('N1','D'): np.array([0.2, 0.3, 0.4]),
    ('N2','L'): np.array([0.11, 0.21, 0.31]),
    ('N2','R'): np.array([0.02, 0.12, 0.22]),
    ('N2','D'): np.array([0.18, 0.28, 0.38, 0.48]),
    ('N3','L'): np.array([0.5, 0.6, 0.7]),
    ('N3','R'): np.array([0.4, 0.45, 0.5]),
    ('N3','D'): np.array([0.05, 0.06]),
}

results_dict, df = compute_threshold_percentages(fa_data)
print('DataFrame:')
print(df)
fig, axes = plot_rejected_threshold_percentages(results_dict, outpath='/home/helios/ubuntu_data/Dual_MesoSPIM/analysis_segmented/R_matrices_segm/rejected_thresholds_test.png')
print('Saved rejected plot to rejected_thresholds_test.png')

