
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from openpyxl.styles.numbers import COLORS

from source.fuse_dual_tomograms import segment_backround_otsu

# =============================================================
# ANATOMICAL MEASUREMENTS FROM EXCEL
# =============================================================
# Path hardcoded al file Excel
EXCEL_PATH = '/home/fra/Pycharm_Projects/whole_heart/data/Anatomical_Measurements_mid_cavity_AVG.xlsx'

# carica lo sheet "medie_fra_campioni" dal file Excel, la prima colonna contiente i due gruppi
wall_thickness = pd.read_excel(EXCEL_PATH, sheet_name='medie_fra_campioni', skiprows=1, index_col=0)
# Conversione dei nomi delle colonne in numeri interi, se necessario
wall_thickness.columns = wall_thickness.columns.astype(int)
# =============================================================

# =============================================================
# LOAD DATAFRAMES OF AMOUNT OF COLLAGEN FROM PICKLE FILES
# =============================================================

# load dataframes from these paths:
CTRL_perc_CF_path = '/home/fra/Pycharm_Projects/whole_heart/data/average_dict_radii_theta/CTRL_perc_CF.pkl'
CTRL_perc_NCF_path = '/home/fra/Pycharm_Projects/whole_heart/data/average_dict_radii_theta/CTRL_perc_NCF.pkl'
PATHOL_perc_CF_path = '/home/fra/Pycharm_Projects/whole_heart/data/average_dict_radii_theta/PATHOL_perc_CF.pkl'
PATHOL_perc_NCF_path = '/home/fra/Pycharm_Projects/whole_heart/data/average_dict_radii_theta/PATHOL_perc_NCF.pkl'

# load the dataframes using pickle
CTRL_perc_CF = pd.read_pickle(CTRL_perc_CF_path)
CTRL_perc_NCF = pd.read_pickle(CTRL_perc_NCF_path)
PATHOL_perc_CF = pd.read_pickle(PATHOL_perc_CF_path)
PATHOL_perc_NCF = pd.read_pickle(PATHOL_perc_NCF_path)

# # print the dataframes
# print("CTRL_perc_CF:\n", CTRL_perc_CF)
# print("CTRL_perc_NCF:\n", CTRL_perc_NCF)
# print("PATHOL_perc_CF:\n", PATHOL_perc_CF)
# print("PATHOL_perc_NCF:\n", PATHOL_perc_NCF)

# somma CTRL_perc_CF and CTRL_perc_NCF
CTRL_perc_fibrosis = CTRL_perc_CF + CTRL_perc_NCF
# somma PATHOL_perc_CF and PATHOL_perc_NCF
PATHOL_perc_fibrosis = PATHOL_perc_CF + PATHOL_perc_NCF

# save as excel files
# CTRL_perc_fibrosis.to_excel('/home/fra/Pycharm_Projects/whole_heart/data/CTRL_perc_total_fibrosis.xlsx')
# PATHOL_perc_fibrosis.to_excel('/home/fra/Pycharm_Projects/whole_heart/data/PATHOL_perc_total_fibrosis.xlsx')

# SOMMA PER TUTTI I RAGGI
# somma le colonne di CTRL_perc_fibrosis
CTRL_perc_fibrosis_sum = CTRL_perc_fibrosis.sum(axis=0)
# somma le colonne di PATHOL_perc_fibrosis
PATHOL_perc_fibrosis_sum = PATHOL_perc_fibrosis.sum(axis=0)

# save as excel files
# CTRL_perc_fibrosis_sum.to_excel('/home/fra/Pycharm_Projects/whole_heart/data/CTRL_perc_total_fibrosis_sum_on_radii.xlsx')
# PATHOL_perc_fibrosis_sum.to_excel('/home/fra/Pycharm_Projects/whole_heart/data/PATHOL_perc_total_fibrosis_sum_sum_on_radii.xlsx')
# =============================================================

# Print the results

print("\n\n *** wall_thickness:", wall_thickness)
print("\n\n *** CTRL_perc_fibrosis_sum:", CTRL_perc_fibrosis_sum)
print("\n\n *** PATHOL_perc_fibrosis_sum:", PATHOL_perc_fibrosis_sum)


#===========================================================
# stampa i nomi delle colonne di CTRL_perc_fibrosis_sum
print("\n\n *** CTRL_perc_fibrosis_sum columns:", CTRL_perc_fibrosis_sum.index.tolist())



