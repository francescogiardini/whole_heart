import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import re


def extract_fa_from_valid_cells(path_file):
    """Estrae i valori di 'fa' dalle celle con 'cell_info' == True."""
    try:
        # Carica il file numpy
        R = np.load(path_file)

        # Applica la condizione per filtrare i dati
        fa_values = R['fa'][R['cell_info']]

        return fa_values
    except Exception as e:
        print(f"Errore durante l'estrazione: {e}")
        return None


def carica_fa_da_file(path_file):
    """Carica il campo 'fa' da un file .npy"""
    try:
        R = np.load(path_file)
        return R['fa']
    except Exception as e:
        print(f"Errore nel caricamento di {path_file}: {e}")
        return None


def extract_fa_from_all_R_files(path_cartella):
    """
    Restituisce un dizionario (sample, misura) -> FA array
    """
    fa_data = dict()

    for nome_file in os.listdir(path_cartella):
        if nome_file.endswith('.npy') and nome_file.startswith('R_'):
            path_completo = os.path.join(path_cartella, nome_file)
            try:
                # extract name (number) of the sample
                parts = nome_file.split('_')
                measure_name = parts[1]  # es: D1, L3, R11...
                region = measure_name[0]  # 'L', 'R', 'D'
                sample_name = 'N' + measure_name[1:]  # '1' -> 'N1', '7'->'N7', '11'->'N11'...
            except Exception as e:
                print(f"Nome file non riconosciuto: {nome_file}")
                continue

            # fa = carica_fa_da_file(path_completo)
            fa = extract_fa_from_valid_cells(path_completo)
            if fa is not None:
                fa_data[(sample_name, region)] = fa

    return fa_data


def calcola_media_std_valori_validi(fa):
    """Calcola media e deviazione standard ignorando valori inf e NaN."""
    # Filtra i valori validi (non NaN e non inf)
    fa_valid = fa[~np.isnan(fa) & ~np.isinf(fa)].astype(np.float64)

    # togli i valori uguali a zero
    fa_valid_nozero = fa_valid[fa_valid != 0]

    # calcola il numero di elementi contenuti in fa_validi
    num_valid_values = len(fa_valid)
    num_nonzero_values = len(fa_valid_nozero)

    print(f"[DEBUG] Max FA value: {np.max(fa_valid)}")
    print(f"[DEBUG] dtype FA: {fa_valid.dtype}")

    # Calcola media e deviazione standard sui valori validi
    mean_fa = np.mean(fa_valid)
    mean_nonzero_fa = np.mean(fa_valid_nozero)

    std_fa = np.std(fa_valid)
    std_nonzero_fa = np.std(fa_valid_nozero)

    return mean_fa, std_fa, num_valid_values, mean_nonzero_fa, std_nonzero_fa, num_nonzero_values


def aggiorna_excel_con_fa(path_excel, fa_data, output_excel_path):
    df = pd.read_excel(path_excel)

    # Nuove colonne per media e std
    mean_col = []
    std_col = []
    num_valid_fa_col = []
    mean_nonzerofa_col = []
    std_nonzerofa_col = []
    num_nonzero_fa_col = []

    print("\nRisultati FA per campione e misura:")
    print("-" * 40)

    for _, row in df.iterrows():
        sample = row['sample'] # es: N1, N2, N3
        region = row['region']  # es: L, R, D

        key = (sample, region)
        fa = fa_data.get(key)

        if fa is not None:
            mean_fa, std_fa, num_valid_values, mean_nonzero_fa, std_nonzero_fa, num_nonzero_values = calcola_media_std_valori_validi(fa)
            mean_col.append(mean_fa)
            std_col.append(std_fa)
            num_valid_fa_col.append(num_valid_values)
            mean_nonzerofa_col.append(mean_nonzero_fa)
            std_nonzerofa_col.append(std_nonzero_fa)
            num_nonzero_fa_col.append(num_nonzero_values)
            print(f"{sample} - {region}: FA mean = {mean_fa:.4f}, std = {std_fa:.4f}, NumValues = {num_valid_values}")
            print(f"Non-zero FA mean = {mean_nonzero_fa:.4f}, std = {std_nonzero_fa:.4f}, NumNonZeroValues = {num_nonzero_values}")
        else:
            print(f"{sample} - {region}: FA non trovata")
            mean_col.append(np.nan)
            std_col.append(np.nan)
            num_valid_fa_col.append(0)
            mean_nonzerofa_col.append(np.nan)
            std_nonzerofa_col.append(np.nan)
            num_nonzero_fa_col.append(0)

    df['FA_mean'] = mean_col
    df['FA_std'] = std_col
    df['NumValidFA'] = num_valid_fa_col
    df['FA_mean_nonzero'] = mean_nonzerofa_col
    df['FA_std_nonzero'] = std_nonzerofa_col
    df['NumNonZeroFA'] = num_nonzero_fa_col

    # Salva nuovo Excel
    df.to_excel(output_excel_path, index=False)
    print(f"\nExcel aggiornato salvato in: {output_excel_path}")

# Funzione di ordinamento personalizzata per i campioni
# Esempio: 'N1', 'N2', 'N10' -> ordina come 1, 2, 10
def sample_key(s):
    m = re.search(r'\d+', s)
    return int(m.group()) if m else s


def compute_threshold_percentages(fa_data, thresholds=None, exclude_zeros=True):
    """
    Calcola per ogni coppia (sample, region) la percentuale di valori FA che
    superano ciascuna soglia nella lista `thresholds`.

    Parametri:
    - fa_data: dict with keys (sample, region) -> numpy array of FA values
    - thresholds: iterable di soglie (float). Default: [0.1,0.2,0.3,0.4,0.5]
    - exclude_zeros: se True esclude i valori esatti 0 dal denominatore

    Ritorna:
    - results_dict: dict mapping (sample,region) -> {threshold: percentuale}
      (percentuale è un float tra 0 e 100, oppure np.nan se non ci sono valori validi)
    - df: pandas.DataFrame con indice MultiIndex (sample, region) e colonne
      'pct_over_{threshold}' per ogni soglia, ordinato per sample (usando sample_key)
    """
    if thresholds is None:
        # thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    results = {}
    rows = []

    for key, fa in fa_data.items():
        # key è una tupla (sample, region)
        sample, region = key
        fa_arr = np.asarray(fa)

        # valori validi: non NaN e non inf
        valid = fa_arr[~np.isnan(fa_arr) & ~np.isinf(fa_arr)]
        if exclude_zeros:
            valid = valid[valid != 0]

        total = valid.size
        th_dict = {}
        row = {'sample': sample, 'region': region}

        for t in thresholds:
            if total == 0:
                pct = np.nan
            else:
                pct = 100.0 * np.count_nonzero(valid > t) / total
            th_dict[t] = pct
            # col name compatible as column in DataFrame
            col_name = f"pct_over_{str(t).replace('.','_')}"
            row[col_name] = pct

        results[key] = th_dict
        rows.append(row)

    # costruisci DataFrame e ordina per sample (usando sample_key) e poi region
    if len(rows) == 0:
        df = pd.DataFrame(columns=[f"pct_over_{str(t).replace('.','_')}" for t in thresholds])
        df.index.names = ['sample', 'region']
        return results, df

    df = pd.DataFrame(rows)

    # add a sortable key for samples (numeric extraction)
    df['sample_sort'] = df['sample'].astype(str).apply(sample_key)
    df = df.sort_values(['sample_sort', 'region']).drop(columns=['sample_sort'])
    df = df.set_index(['sample', 'region'])

    return results, df


def plot_fa_distributions_by_sample(fa_data, bins=50, outpath=None):
    """
    Crea una figura con istogrammi dei valori FA per ogni misura (L/R/D) e campione.
    Ogni colonna è un campione, ogni riga una misura (L, R, D).
    """
    # Ordina i campioni
    samples = sorted({key[0] for key in fa_data}, key=sample_key)
    regions = ['L', 'R', 'D']

    num_rows = len(regions)
    num_cols = len(samples)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows), sharex=True)
    fig.suptitle('Distribuzioni dei valori di FA', fontsize=16)

    for col, sample in enumerate(samples):
        for row, region in enumerate(regions):
            ax = axes[row, col] if num_rows > 1 else axes[col]
            key = (sample, region)  # esempio: ('N1', 'L'), ('N1', 'R'), ('N1', 'D'), ('N2', 'L'), etc.

            if key in fa_data:
                fa_values = fa_data[key]
                fa_values = fa_values.astype(np.float64)
                fa_valid = fa_values[~np.isnan(fa_values) & ~np.isinf(fa_values) & (fa_values > 0)]
                ax.hist(fa_valid, bins=bins, range=(0, 1), color='steelblue', alpha=0.8)
            else:
                ax.text(0.5, 0.5, 'Dati mancanti', ha='center', va='center', fontsize=9)

            if row == 0:
                ax.set_title(sample)
            if col == 0:
                ax.set_ylabel(f"Misura: {region}")

    for ax in axes.flat:
        ax.set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight')
        print(f"Grafico salvato in: {outpath}")
    else:
        print("Nessun percorso di salvataggio specificato per il grafico.")


def plot_threshold_percentages(results_dict, thresholds=None, outpath=None, figsize_per_plot=(4,3)):
    """
    Crea un plot a pannelli: ogni campione ha un subplot; per ogni campione vengono
    plottate le percentuali di valori FA sopra le soglie per le tre regioni (L,R,D)
    come linee con colori diversi (L=red, R=green, D=black).

    Parametri:
    - results_dict: dict mapping (sample, region) -> {threshold: percentuale}
    - thresholds: lista ordinata di soglie (default: [0.1,0.2,0.3,0.4,0.5])
    - outpath: se fornito, salva la figura in questo percorso (es. 'out.png')
    - figsize_per_plot: tuple (width, height) per singolo subplot (default (4,3))

    Ritorna:
    - fig, axes: oggetti matplotlib
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    # colori per le regioni
    colors = {'L': 'red', 'R': 'green', 'D': 'black'}
    regions = ['L', 'R', 'D']

    # elenco campioni ordinato numericamente
    samples = sorted({k[0] for k in results_dict.keys()}, key=sample_key)
    n = len(samples)
    if n == 0:
        raise ValueError("results_dict vuoto: niente da plottare")

    # layout subplot
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig_w = ncols * figsize_per_plot[0]
    fig_h = nrows * figsize_per_plot[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    # normalizza axes in lista (flatten) per iterazione uniforme
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for i, sample in enumerate(samples):
        ax = axes_flat[i]
        # raccogli i dizionari per regione per questo sample (per riutilizzo)
        region_values = {r: results_dict.get((sample, r), {}) for r in regions}
        for region in regions:
            region_dict = region_values[region]
            # costruisci lista di y nello stesso ordine delle thresholds (rejected = 100 - pct_over)
            y = []
            for t in thresholds:
                v = region_dict.get(t, np.nan)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    y.append(np.nan)
                else:
                    y.append(100.0 - float(v))
            ax.plot(thresholds, y, marker='o', color=colors.get(region, 'gray'), label=region)

        # disegna segmenti grigi che collegano i tre valori (L,R,D) per ogni threshold
        for t in thresholds:
            yvals = [region_values[r].get(t, np.nan) for r in regions]
            # trasform to rejected and filter NaN
            y_clean = [100.0 - float(y) for y in yvals if not (isinstance(y, float) and np.isnan(y))]
            if len(y_clean) >= 2:
                ymin = min(y_clean)
                ymax = max(y_clean)
                ax.plot([t, t], [ymin, ymax], color='gray', linewidth=1.0, alpha=0.6, zorder=0)

        ax.set_title(sample)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Percentuale (%)')
        ax.set_xticks(thresholds)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(title='Regione')

    # nascondi assi inutilizzati
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight')
        print(f"Grafico thresholds salvato in: {outpath}")

    return fig, axes


def plot_rejected_threshold_percentages(results_dict, thresholds=None, outpath=None, figsize_per_plot=(4,3)):
    """
    Similar to plot_threshold_percentages but plots the percentage of rejected fibers:
    rejected_pct = 100 - pct_over.

    - results_dict: dict mapping (sample, region) -> {threshold: percentuale}
    - thresholds: list of thresholds (default [0.1..0.5])
    - outpath: optional path to save the figure
    - figsize_per_plot: tuple for each subplot size

    Returns fig, axes
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    colors = {'L': 'red', 'R': 'green', 'D': 'black'}
    regions = ['L', 'R', 'D']

    # ordered samples
    try:
        samples = sorted({k[0] for k in results_dict.keys()}, key=sample_key)
    except Exception:
        samples = sorted({k[0] for k in results_dict.keys()})

    n = len(samples)
    if n == 0:
        raise ValueError("results_dict vuoto: niente da plottare")

    # layout
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig_w = ncols * figsize_per_plot[0]
    fig_h = nrows * figsize_per_plot[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    # normalizza axes in lista (flatten) per iterazione uniforme
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for i, sample in enumerate(samples):
        ax = axes_flat[i]
        # raccogli i dizionari per regione per questo sample (per riutilizzo)
        region_values = {r: results_dict.get((sample, r), {}) for r in regions}
        for region in regions:
            region_dict = region_values[region]
            # costruisci lista di y nello stesso ordine delle thresholds (rejected = 100 - pct_over)
            y = []
            for t in thresholds:
                v = region_dict.get(t, np.nan)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    y.append(np.nan)
                else:
                    y.append(100.0 - float(v))
            ax.plot(thresholds, y, marker='o', color=colors.get(region, 'gray'), label=region)

        # disegna segmenti grigi che collegano i tre valori (L,R,D) per ogni threshold
        for t in thresholds:
            yvals = [region_values[r].get(t, np.nan) for r in regions]
            # trasform to rejected and filter NaN
            y_clean = [100.0 - float(y) for y in yvals if not (isinstance(y, float) and np.isnan(y))]
            if len(y_clean) >= 2:
                ymin = min(y_clean)
                ymax = max(y_clean)
                ax.plot([t, t], [ymin, ymax], color='gray', linewidth=1.0, alpha=0.6, zorder=0)

        ax.set_title(sample)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Percentuale fibre scartate (%)')
        ax.set_xticks(thresholds)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(title='Regione')

    # hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight')
        print(f"Grafico rejected thresholds salvato in: {outpath}")

    return fig, axes


def plot_threshold_stats(results_dict, thresholds=None, outpath=None, alpha=0.05, figsize=(12,8)):
    """
    Crea 5 pannelli (uno per ogni threshold). Per ogni threshold plottare uno scatter
    con i valori percentuali che superano quella soglia: x = regione (L,R,D),
    y = percentuale (0-100). Usa colori L=red, R=green, D=black.

    Inoltre traccia la media per ogni regione e, se disponibile scipy, testa la
    gaussianità di ciascuna distribuzione (Shapiro) e — se tutte e tre sono
    gaussiane — esegue t-test (indipendenti) fra le coppie (L-R, L-D, R-D).
    Se il p-value < alpha per una coppia, disegna una linea di confronto con
    un asterisco sul subplot.

    Parametri:
    - results_dict: dict mapping (sample, region) -> {threshold: percentuale}
    - thresholds: lista di soglie (default: [0.1,...,0.5])
    - outpath: percorso per salvare figura (opzionale)
    - alpha: significatività per t-test
    - figsize: dimensione figura complessiva

    Ritorna (fig, axes)
    """
    import math
    try:
        from scipy import stats
        scipy_available = True
    except Exception:
        scipy_available = False

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    colors = {'L': 'red', 'R': 'green', 'D': 'black'}
    regions = ['L', 'R', 'D']

    # estrai campioni ordinati
    try:
        samples = sorted({k[0] for k in results_dict.keys()}, key=sample_key)
    except Exception:
        samples = sorted({k[0] for k in results_dict.keys()})

    # costruisci per ogni threshold e regione la lista delle percentuali attraverso i campioni
    data_by_threshold = {}
    for t in thresholds:
        data_by_threshold[t] = {r: [] for r in regions}

    for (sample, region), th_dict in results_dict.items():
        for t in thresholds:
            val = th_dict.get(t, np.nan)
            if not (val is None or (isinstance(val, float) and np.isnan(val))):
                data_by_threshold[t][region].append(float(val))

    # crea figura con len(thresholds) pannelli
    n = len(thresholds)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    x_positions = { 'L': 0, 'R': 1, 'D': 2 }

    for i, t in enumerate(thresholds):
        ax = axes[i]
        # scatter punti per regione
        max_y = 0
        for region in regions:
            ys = np.array(data_by_threshold[t][region], dtype=float)
            # filtro nan
            ys = ys[~np.isnan(ys)]
            if ys.size == 0:
                continue
            max_y = max(max_y, np.nanmax(ys))
            # jitter sull'asse x per separare i punti
            jitter = (np.random.rand(len(ys)) - 0.5) * 0.15
            xs = np.full_like(ys, x_positions[region], dtype=float) + jitter
            ax.scatter(xs, ys, color=colors.get(region, 'gray'), alpha=0.8, label=region)
            # media e marker
            mean_val = np.nanmean(ys)
            ax.plot([x_positions[region]-0.2, x_positions[region]+0.2], [mean_val, mean_val], color=colors.get(region), linewidth=2)
            ax.scatter([x_positions[region]], [mean_val], color='white', edgecolor=colors.get(region), zorder=5, s=40)

        # per ogni sample, collega i punti delle 3 regioni (L,R,D) con una linea grigia
        for sample in samples:
            yvals = [results_dict.get((sample, r), {}).get(t, np.nan) for r in regions]
            yarr = np.array(yvals, dtype=float)
            mask = ~np.isnan(yarr)
            if np.count_nonzero(mask) >= 2:
                xs_line = [x_positions[r] for idx, r in enumerate(regions) if mask[idx]]
                ys_line = yarr[mask]
                ax.plot(xs_line, ys_line, color='gray', linewidth=0.8, alpha=0.6, zorder=0)

        # labeling
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['L','R','D'])
        ax.set_ylim(0, max(100, math.ceil(max_y + 10)))
        ax.set_ylabel('Percentuale (%)')
        ax.set_title(f'Threshold = {t}')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # statistiche: test gaussianità e t-test se possibile
        stat_texts = []
        region_normals = {}
        for region in regions:
            ys = np.array(data_by_threshold[t][region], dtype=float)
            ys = ys[~np.isnan(ys)]
            if ys.size < 3:
                region_normals[region] = False
                stat_texts.append(f"{region}: n={len(ys)}")
                continue
            # se la varianza è zero (tutti i valori uguali) non eseguire lo Shapiro
            if np.nanstd(ys) == 0:
                region_normals[region] = False
                stat_texts.append(f"{region}: constant (std=0)")
                continue
            if scipy_available:
                # Shapiro-Wilk test (safe per campioni piccoli). Proteggiamo dalle warning
                try:
                    with __import__('warnings').catch_warnings():
                        __import__('warnings').simplefilter('ignore')
                        w, p = stats.shapiro(ys)
                    is_normal = (p is not None) and (p > alpha)
                    region_normals[region] = bool(is_normal)
                    stat_texts.append(f"{region}: p_shapiro={p:.3f}")
                except Exception:
                    region_normals[region] = False
                    stat_texts.append(f"{region}: shapiro_err")
            else:
                region_normals[region] = False
                stat_texts.append(f"{region}: scipy_missing")

        # se possibile, esegui test fra coppie: se tutte gaussiane -> t-test, altrimenti Mann-Whitney U
        sig_pairs = []
        if scipy_available:
            pairs = [('L','R'), ('L','D'), ('R','D')]
            used_y = max(100, math.ceil(max_y + 10))
            step = used_y * 0.06
            cur_offset = 1
            for (a,b) in pairs:
                ya = np.array(data_by_threshold[t][a], dtype=float); ya = ya[~np.isnan(ya)]
                yb = np.array(data_by_threshold[t][b], dtype=float); yb = yb[~np.isnan(yb)]
                if len(ya) < 2 or len(yb) < 2:
                    continue

                pval = None
                test_name = None
                try:
                    if region_normals.get(a, False) and region_normals.get(b, False):
                        # t-test (parametrico)
                        test_name = 'ttest'
                        _, pval = stats.ttest_ind(ya, yb, equal_var=False, nan_policy='omit')
                    else:
                        # Mann-Whitney U (non-parametrico)
                        test_name = 'mannwhitney'
                        # use two-sided
                        try:
                            u_stat, pval = stats.mannwhitneyu(ya, yb, alternative='two-sided')
                        except TypeError:
                            # older scipy might not accept 'alternative'
                            u_stat, pval = stats.mannwhitneyu(ya, yb)
                except Exception:
                    pval = None

                if pval is not None and (not np.isnan(pval)) and pval < alpha:
                    # disegna linea di confronto
                    x1 = x_positions[a]
                    x2 = x_positions[b]
                    y_line = used_y - cur_offset * step
                    ax.plot([x1, x1, x2, x2], [y_line-1, y_line, y_line, y_line-1], color='k', linewidth=1.0)
                    ax.text((x1+x2)/2, y_line + 0.5, '*', ha='center', va='bottom', fontsize=14)
                    cur_offset += 1
                    sig_pairs.append(((a,b), pval, test_name))
                # aggiungi info test al testo di debug
                if pval is None or np.isnan(pval):
                    stat_texts.append(f"{a}-{b}: test_err")
                else:
                    stat_texts.append(f"{a}-{b}: {test_name} p={pval:.3f}")
        else:
            stat_texts.append('scipy missing: no tests')

        # mostra testo con risultati dei test in basso a sinistra del subplot
        ax.text(0.01, 0.01, '\n'.join(stat_texts), transform=ax.transAxes, fontsize=8, va='bottom')

    plt.tight_layout()
    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight')
        print(f"Grafico statistico thresholds salvato in: {outpath}")

    return fig, axes


def plot_rejected_threshold_stats(results_dict, thresholds=None, outpath=None, alpha=0.05, figsize=(12,8)):
    """
    Simile a plot_threshold_stats, ma lavora sulle percentuali di fibre scartate:
    rejected = 100 - pct_over_threshold.

    Crea un pannello per ogni threshold (default 5). In ogni pannello:
    - scatter dei valori rejected per regione (L,R,D) con jitter
    - linea della media per regione
    - test accoppiati (paired t-test se differenze normali, altrimenti Wilcoxon)
      sulle coppie L-R, L-D, R-D, segnando con '*' le coppie significative.

    Parametri:
    - results_dict: dict (sample,region) -> {threshold: pct_over}
    - thresholds: lista soglie
    - outpath: se fornito salva la figura
    - alpha: soglia di significatività
    - figsize: dimensione figura

    Ritorna: fig, axes
    """
    import math
    try:
        from scipy import stats
        scipy_available = True
    except Exception:
        scipy_available = False

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    colors = {'L': 'red', 'R': 'green', 'D': 'black'}
    regions = ['L', 'R', 'D']

    # estrai campioni ordinati
    try:
        samples = sorted({k[0] for k in results_dict.keys()}, key=sample_key)
    except Exception:
        samples = sorted({k[0] for k in results_dict.keys()})

    # costruisci mapping threshold -> region -> {sample: rejected_value}
    data_by_threshold = {}
    for t in thresholds:
        data_by_threshold[t] = {r: {} for r in regions}

    for (sample, region), th_dict in results_dict.items():
        for t in thresholds:
            v = th_dict.get(t, np.nan)
            if not (v is None or (isinstance(v, float) and np.isnan(v))):
                rejected = 100.0 - float(v)
                data_by_threshold[t][region][sample] = rejected

    # crea figura
    n = len(thresholds)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    x_positions = {'L': 0, 'R': 1, 'D': 2}

    for i, t in enumerate(thresholds):
        ax = axes[i]
        max_y = 0
        # plot punti e medie
        for region in regions:
            samples_vals = data_by_threshold[t][region]
            ys = np.array([v for v in samples_vals.values()], dtype=float)
            if ys.size == 0:
                continue
            ys = ys[~np.isnan(ys)]
            max_y = max(max_y, np.nanmax(ys))
            jitter = (np.random.rand(len(ys)) - 0.5) * 0.15
            xs = np.full_like(ys, x_positions[region], dtype=float) + jitter
            ax.scatter(xs, ys, color=colors.get(region, 'gray'), alpha=0.8, label=region)
            mean_val = np.nanmean(ys)
            ax.plot([x_positions[region]-0.2, x_positions[region]+0.2], [mean_val, mean_val], color=colors.get(region), linewidth=2)
            ax.scatter([x_positions[region]], [mean_val], color='white', edgecolor=colors.get(region), zorder=5, s=40)

        # labeling
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['L','R','D'])
        ax.set_ylim(0, max(100, math.ceil(max_y + 10)))
        ax.set_ylabel('Percentuale fibre scartate (%)')
        ax.set_title(f'Threshold = {t}')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # statistiche accoppiate
        stat_texts = []
        if not scipy_available:
            stat_texts.append('scipy missing: no tests')
        else:
            pairs = [('L','R'), ('L','D'), ('R','D')]
            used_y = max(100, math.ceil(max_y + 10))
            step = used_y * 0.06
            cur_offset = 1
            for (a,b) in pairs:
                samples_a = data_by_threshold[t][a]
                samples_b = data_by_threshold[t][b]
                paired_samples = [s for s in samples if (s in samples_a and s in samples_b)]
                ya = np.array([samples_a[s] for s in paired_samples], dtype=float)
                yb = np.array([samples_b[s] for s in paired_samples], dtype=float)

                stat_texts.append(f"{a}-{b}: n_pairs={len(paired_samples)}")
                if len(paired_samples) < 2:
                    stat_texts.append(f"{a}-{b}: insufficient pairs")
                    continue

                diffs = ya - yb
                if np.nanstd(diffs) == 0:
                    stat_texts.append(f"{a}-{b}: no difference (constant)")
                    continue

                # test normalità sulle differenze
                is_normal = False
                p_shapiro = None
                try:
                    if len(diffs) >= 3:
                        with __import__('warnings').catch_warnings():
                            __import__('warnings').simplefilter('ignore')
                            w, p_shapiro = stats.shapiro(diffs)
                        is_normal = (p_shapiro is not None) and (p_shapiro > alpha)
                        stat_texts.append(f"{a}-{b}: p_shapiro_diff={p_shapiro:.3f}")
                    else:
                        stat_texts.append(f"{a}-{b}: shapiro skipped (n<3)")
                except Exception:
                    stat_texts.append(f"{a}-{b}: shapiro_err")

                # scegli test accoppiato
                pval = None
                test_name = None
                try:
                    if is_normal:
                        test_name = 'paired_t'
                        _, pval = stats.ttest_rel(ya, yb, nan_policy='omit')
                    else:
                        test_name = 'wilcoxon'
                        try:
                            stat_w, pval = stats.wilcoxon(ya, yb)
                        except TypeError:
                            stat_w, pval = stats.wilcoxon(ya - yb)
                except Exception:
                    pval = None

                if pval is not None and (not np.isnan(pval)) and pval < alpha:
                    x1 = x_positions[a]
                    x2 = x_positions[b]
                    y_line = used_y - cur_offset * step
                    ax.plot([x1, x1, x2, x2], [y_line-1, y_line, y_line, y_line-1], color='k', linewidth=1.0)
                    ax.text((x1+x2)/2, y_line + 0.5, '*', ha='center', va='bottom', fontsize=14)
                    cur_offset += 1

                if pval is None or np.isnan(pval):
                    stat_texts.append(f"{a}-{b}: test_err")
                else:
                    stat_texts.append(f"{a}-{b}: {test_name} p={pval:.3f}")

        ax.text(0.01, 0.01, '\n'.join(stat_texts), transform=ax.transAxes, fontsize=8, va='bottom')

    plt.tight_layout()
    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight')
        print(f"Grafico rejected thresholds (stats) salvato in: {outpath}")

    return fig, axes


def save_results_to_excel(results_dict, outpath, thresholds=None, regions=('L','R','D'), include_rejected=False):
    """
    Salva `results_dict` in un file Excel `outpath`.
    - results_dict: dict[(sample, region)] -> {threshold: pct_over}
    - thresholds: iterable di soglie da esportare (se None vengono dedotte dai dizionari)
    - regions: ordine delle colonne (default ('L','R','D'))
    - include_rejected: se True aggiunge per ogni regione la colonna 'rejected' = 100 - pct_over

    Crea un foglio per ogni threshold (nome: 'thr_0_2' per threshold 0.2). Ritorna il percorso scritto.
    """
    if not results_dict:
        raise ValueError("results_dict è vuoto")

    # infer thresholds se non forniti
    if thresholds is None:
        th_set = set()
        for thdict in results_dict.values():
            try:
                th_set.update(thdict.keys())
            except Exception:
                continue
        thresholds = sorted(th_set)

    # raccogli campioni ordinati usando sample_key definita nel modulo
    try:
        samples = sorted({k[0] for k in results_dict.keys()}, key=sample_key)
    except Exception:
        samples = sorted({k[0] for k in results_dict.keys()})

    # prepara ExcelWriter
    try:
        # specifica engine openpyxl se disponibile
        writer = pd.ExcelWriter(outpath, engine='openpyxl')
    except Exception:
        writer = pd.ExcelWriter(outpath)

    with writer:
        for t in thresholds:
            rows = []
            for s in samples:
                row = {}
                for r in regions:
                    val = results_dict.get((s, r), {}).get(t, np.nan)
                    if val is None:
                        val = np.nan
                    try:
                        valf = float(val)
                    except Exception:
                        valf = np.nan

                    if include_rejected:
                        row[("pct_over", r)] = np.nan if np.isnan(valf) else valf
                        row[("rejected", r)] = np.nan if np.isnan(valf) else (100.0 - valf)
                    else:
                        row[r] = np.nan if np.isnan(valf) else valf
                rows.append(row)

            if include_rejected:
                df = pd.DataFrame(rows, index=samples)
                # assicurati ordine colonne: metriche x regioni
                cols = pd.MultiIndex.from_product([["pct_over", "rejected"], list(regions)])
                df = df.reindex(columns=cols)
            else:
                df = pd.DataFrame(rows, index=samples)
                df = df.reindex(columns=list(regions))

            df.index.name = 'sample'
            # nome foglio sicuro (<=31 chars)
            sheet_name = f"thr_{str(t).replace('.', '_')}"[:31]
            df.to_excel(writer, sheet_name=sheet_name)

    return outpath

def main(parser):
    # Parse command line arguments
    args = parser.parse_args()
    cartella = args.cartella
    input_excel_fpath = args.file_excel

    basepath = os.path.dirname(input_excel_fpath)

    # Verifica se la cartella esiste
    if not os.path.isdir(cartella):
        print(f"Errore: La cartella {cartella} non esiste.")
        return
    # Verifica se il file Excel esiste
    if not os.path.isfile(input_excel_fpath):
        print(f"Errore: Il file Excel {input_excel_fpath} non esiste.")
        return

    # ======================================
    # Raed FA values from .npy files in the specified folder
    # Dict (sample, measure) -> FA array
    # ======================================
    fa_data = extract_fa_from_all_R_files(cartella)
    # example of fa_data key: ('N1', 'L'), ('N3', 'R'), ('N7', 'D'), etc.

    # ======================================
    # Aggiorna il file Excel con i dati FA
    # ======================================
    output_excel_path = os.path.join(basepath, "Output_with_FA_aggiornato.xlsx")
    aggiorna_excel_con_fa(input_excel_fpath, fa_data, output_excel_path)

    # ======================================
    # Plot distributions of FA values by sample and measure
    # ======================================
    plot_fa_distributions_by_sample(fa_data, bins=50, outpath=os.path.join(basepath, 'FA_dist.png'))


    # ======================================
    # Compute threshold percentages
    # ======================================
    # thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    results_dict, df_thresholds = compute_threshold_percentages(fa_data, thresholds=thresholds, exclude_zeros=True)

    # ======================================
    # Plot threshold percentages
    # ======================================
    plot_threshold_percentages(results_dict, thresholds=thresholds,
                               outpath=os.path.join(basepath, 'thresholds_plot.png'))

    # ======================================
    # Plot threshold statistics
    # ======================================
    plot_threshold_stats(results_dict, thresholds=thresholds,
                         outpath=os.path.join(basepath, 'thresholds_stats_plot.png'))

    # ======================================
    # Plot rejected threshold percentages
    # ======================================
    plot_rejected_threshold_percentages(results_dict, thresholds=thresholds,
                                       outpath=os.path.join(basepath, 'rejected_thresholds_plot.png'))

    # ======================================
    # Plot rejected threshold statistics
    # ======================================
    plot_rejected_threshold_stats(results_dict, thresholds=thresholds,
                                  outpath=os.path.join(basepath, 'rejected_thresholds_stats_plot.png'))

    # ======================================
    # Salva i risultati in Excel
    # ======================================
    save_results_to_excel(results_dict, outpath=os.path.join(basepath, 'FA_stats_for_thresholds.xlsx'),
                          thresholds=thresholds, include_rejected=True)



if __name__ == "__main__":
    print("Analisi dei campi FA e aggiornamento Excel")
    parser = argparse.ArgumentParser(description="Analisi FA e aggiornamento Excel")
    parser.add_argument("cartella", help="Path della cartella con i file .npy")
    parser.add_argument("file_excel", help="Path del file Excel di input(.xlsx)")

    main(parser)
