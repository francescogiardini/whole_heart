import numpy as np
from analyze_R_matrices import compute_threshold_percentages

# esempio di fa_data con diversi casi
fa_data = {
    ('N1','L'): np.array([0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, np.nan, np.inf]),
    ('N2','R'): np.array([0.12, 0.22, 0.32, 0.42, 0.52]),
    ('N10','D'): np.array([0, 0, 0, np.nan]),
}

results, df = compute_threshold_percentages(fa_data)
print('Results dict:')
for k, v in results.items():
    print(k, v)

print('\nDataFrame:')
print(df)

