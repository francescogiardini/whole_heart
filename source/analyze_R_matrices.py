import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt


def estrai_fa_condizionato(path_file):
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


def estrai_fa_da_cartella(path_cartella):
    """
    Restituisce un dizionario (sample, misura) -> FA array
    """
    fa_data = dict()

    for nome_file in os.listdir(path_cartella):
        if nome_file.endswith('.npy') and nome_file.startswith('R_'):
            path_completo = os.path.join(path_cartella, nome_file)
            try:
                parts = nome_file.split('_')
                tipo = parts[1]  # es: D1
                misura = tipo[0]  # 'L', 'R', 'D'
                campione = 'N' + tipo[1]  # '1' -> 'N1'
            except Exception as e:
                print(f"Nome file non riconosciuto: {nome_file}")
                continue

            # fa = carica_fa_da_file(path_completo)
            fa = estrai_fa_condizionato(path_completo)
            if fa is not None:
                fa_data[(campione, tipo)] = fa

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
        sample = row['sample']
        subfolder = row['subfolder']  # es: L1, R3, D8

        key = (sample, subfolder)
        fa = fa_data.get(key)

        if fa is not None:
            mean_fa, std_fa, num_valid_values, mean_nonzero_fa, std_nonzero_fa, num_nonzero_values = calcola_media_std_valori_validi(fa)
            mean_col.append(mean_fa)
            std_col.append(std_fa)
            num_valid_fa_col.append(num_valid_values)
            mean_nonzerofa_col.append(mean_nonzero_fa)
            std_nonzerofa_col.append(std_nonzero_fa)
            num_nonzero_fa_col.append(num_nonzero_values)
            print(f"{sample} - {subfolder}: FA mean = {mean_fa:.4f}, std = {std_fa:.4f}, NumValues = {num_valid_values}")
            print(f"Non-zero FA mean = {mean_nonzero_fa:.4f}, std = {std_nonzero_fa:.4f}, NumNonZeroValues = {num_nonzero_values}")
        else:
            print(f"{sample} - {subfolder}: FA non trovata")
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


def plot_fa_distributions_by_sample(fa_data, bins=50, outpath=None):
    """
    Crea una figura con istogrammi dei valori FA per ogni misura (L/R/D) e campione.
    Ogni colonna è un campione, ogni riga una misura (L, R, D).
    """
    # Ordina i campioni
    samples = sorted({key[0] for key in fa_data})
    measures = ['L', 'R', 'D']

    num_rows = len(measures)
    num_cols = len(samples)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows), sharex=True)
    fig.suptitle('Distribuzioni dei valori di FA', fontsize=16)

    for col, sample in enumerate(samples):
        for row, measure in enumerate(measures):
            ax = axes[row, col] if num_rows > 1 else axes[col]
            sample_id = col + 1  # per avere L1, R1, D1...
            key = (sample, f"{measure}{sample_id}")  # esempio: ('N1', 'L1'), ('N1', 'R1')...

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
                ax.set_ylabel(f"Misura: {measure}")

    for ax in axes.flat:
        ax.set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight')
        print(f"Grafico salvato in: {outpath}")
    else:
        print("Nessun percorso di salvataggio specificato per il grafico.")


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
    fa_data = estrai_fa_da_cartella(cartella)

    # ======================================
    # Aggiorna il file Excel con i dati FA
    # ======================================
    # output_excel_path = os.path.join(basepath, "Output_with_FA_aggiornato.xlsx")
    # aggiorna_excel_con_fa(input_excel_fpath, fa_data, output_excel_path)

    # ======================================
    # Plot distributions of FA values by sample and measure
    # ======================================
    plot_fa_distributions_by_sample(fa_data, bins=50, outpath=os.path.join(basepath, 'FA_dist.png'))

if __name__ == "__main__":
    print("Analisi dei campi FA e aggiornamento Excel")
    parser = argparse.ArgumentParser(description="Analisi FA e aggiornamento Excel")
    parser.add_argument("cartella", help="Path della cartella con i file .npy")
    parser.add_argument("file_excel", help="Path del file Excel di input(.xlsx)")

    main(parser)
