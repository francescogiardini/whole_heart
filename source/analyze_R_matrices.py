import os
import numpy as np
import pandas as pd
import argparse


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
    fa_validi = fa[~np.isnan(fa) & ~np.isinf(fa)]

    # calcola il numero di elementi contenuti in fa_validi
    num_valid_values = len(fa_validi)

    # Calcola media e deviazione standard sui valori validi
    mean_val = np.mean(fa_validi)

    std_val = np.std(fa_validi)

    return mean_val, std_val, num_valid_values


def aggiorna_excel_con_fa(path_excel, fa_data, output_excel_path):
    df = pd.read_excel(path_excel)

    # Nuove colonne per media e std
    mean_col = []
    std_col = []
    num_valid_fa_col = []

    print("\nRisultati FA per campione e misura:")
    print("-" * 40)

    for _, row in df.iterrows():
        sample = row['sample']
        subfolder = row['subfolder']  # es: L1, R3, D8

        key = (sample, subfolder)
        fa = fa_data.get(key)

        if fa is not None:
            mean_val, std_val, num_valid_values = calcola_media_std_valori_validi(fa)
            mean_col.append(mean_val)
            std_col.append(std_val)
            num_valid_fa_col.append(num_valid_values)
            print(f"{sample} - {subfolder}: FA mean = {mean_val:.4f}, std = {std_val:.4f}, NumValues = {num_valid_values}")
        else:
            print(f"{sample} - {subfolder}: FA non trovata")
            mean_col.append(np.nan)
            std_col.append(np.nan)
            num_valid_fa_col.append(0)

    df['FA_mean'] = mean_col
    df['FA_std'] = std_col
    df['NumValidFA'] = num_valid_fa_col


    # Salva nuovo Excel
    df.to_excel(output_excel_path, index=False)
    print(f"\nExcel aggiornato salvato in: {output_excel_path}")

if __name__ == "__main__":
    print("Analisi dei campi FA e aggiornamento Excel")
    parser = argparse.ArgumentParser(description="Analisi FA e aggiornamento Excel")
    parser.add_argument("cartella", help="Path della cartella con i file .npy")
    parser.add_argument("file_excel", help="Path del file Excel (.xlsx)")
    args = parser.parse_args()

    # Estrazione dati FA da .npy
    fa_data = estrai_fa_da_cartella(args.cartella)

    # Aggiorna il file Excel e stampa risultati
    output_path = args.file_excel.replace(".xlsx", "_updated.xlsx")
    aggiorna_excel_con_fa(args.file_excel, fa_data, output_path)
    print("\nAnalisi completata!")