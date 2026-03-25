import numpy as np
from analyze_R_matrices import compute_threshold_percentages, plot_threshold_stats

# costruisco un piccolo fa_data di esempio con più campioni
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

thresholds = [0.1,0.2,0.3,0.4,0.5]
results_dict, df = compute_threshold_percentages(fa_data, thresholds=thresholds, exclude_zeros=True)
print('Computed df:')
print(df)

fig, axes = plot_threshold_stats(results_dict, thresholds=thresholds, outpath='/home/helios/ubuntu_data/Dual_MesoSPIM/analysis_segmented/R_matrices_segm/thresholds_stats_test.png')
print('Saved plot to thresholds_stats_test.png')

