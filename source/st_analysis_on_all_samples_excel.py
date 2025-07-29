# run st_analysis.py on all the samples in the input basepath
# example:
# .../basepath/N1/ <- {N1.tif, parameters.txt}
# .../basepath/N2/ <- {N2.tif, parameters.txt}
# .../basepath/N2/ <- {N3.tif, parameters.txt}
# > python disarray_on_all -s /.../basepath   -> run st_analysis.py -s /.../N1; st_analysis.py -s /.../N1 etc.

import argparse
import os
import subprocess
import pandas as pd


class Bcolors:
    VERB = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main(parser):

    # args
    args               = parser.parse_args()
    excel_path         = args.excel_path[0]
    column_index       = args.column_index[0]
    start_raw          = args.row_index[0]
    _disarray_analysis = args.disarray
    _skip_st_analysis  = args.skip_st_analysis
    _plot_quiver       = args.plot_quiver

    # open excel file and read the path of the samples
    df = pd.read_excel(excel_path, header=None)

    # start read the path of the samples from the specified column and row
    source_paths_list = df.iloc[int(start_raw):, int(column_index)]

    print(Bcolors.WARNING + '\n\n**************** Start Fibers Analysis on all Samples in :' + Bcolors.ENDC)
    print('- source_path : ', os.path.basename(excel_path))
    print('- list of samples:')
    for tiffpath in source_paths_list:
        print('   -', tiffpath)
    print('- [optional] Perform Disarray analysis: ', _disarray_analysis)
    print('- [optional] Skip Structure Tensor Analysis: ', _skip_st_analysis)
    print()

    # run analisys on each sample
    for tiffpath in source_paths_list:

        tiffname     = os.path.basename(tiffpath)
        fldrpath = os.path.dirname(tiffpath)

        print(Bcolors.WARNING + '\n Start the analysis on: Sample {}\n'.format(tiffname) + Bcolors.ENDC)

        # parameters file on the current path
        par_fnames = [p for p in os.listdir(fldrpath) if p.startswith('parameters')]
        parampath  = os.path.join(fldrpath, par_fnames[0]) if par_fnames else None

        if not _skip_st_analysis:

            if _plot_quiver:
                # perform analsys calling the sub-script and plot quivers
                os.system('python st_analysis.py -s {} -p {} -q'.format(tiffpath, par_fnames[0]))
            else:
                # perform analsys calling the sub-script
                os.system('python st_analysis.py -s {} -p {}'.format(tiffpath, par_fnames[0]))

        if _disarray_analysis is True:

            # search file numpy R in the current path
            R_fnames = [r for r in os.listdir(fldrpath) if r.startswith('R_') and r.endswith('.npy')]

            # check if there is only one R
            if len(R_fnames) == 1:
                R_path = os.path.join(fldrpath, R_fnames[0])

                # perform disarray analsys calling the sub-script
                os.system('python local_disarray_by_R.py -r {} -p {}'.format(R_path, par_fnames[0]))

    print(Bcolors.WARNING + '\n Finish to analyze all samples\n' + Bcolors.ENDC)
    return None

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run st_analysys on all path in the excel file')
    parser.add_argument('-x', '--excel_path', nargs='+', required=True,
                        help='Perform st_analysis on all samples in input path.')
    # add column index ad parameter
    parser.add_argument('-c', '--column_index', nargs='+', required=True,
                        help='Index of the column in the excel file to read the path of the samples.')
    #  add row index as parameter
    parser.add_argument('-r', '--row_index', nargs='+', required=True,
                        help='Index of the row in the excel file to read the path of the samples.')
    parser.add_argument('-d', '--disarray', action='store_true', default=False, required=False,
                        help='if passed, perform disarray analysis on all samples after the st_analysis.py')
    parser.add_argument('-s', '--skip-st-analysis', action='store_true', default=False, required=False,
                        help='if passed, the script skip the st_analysis.')
    parser.add_argument('-q', '--plot-quiver', action='store_true', default=False, dest='plot_quiver',
                           help='run "plot_quiver_on_mosaic_frame" at the end of the script')

    main(parser)

    # TODO:
    '''
    - --- devo fare un porgramma solo che fa la STA! non ha senso Whole_heart e pig_analysis separato... decidere quale usare e pulire quello 
    - sistemare st_analysi_on_all_samples per prendere anche una lista di path oltre che la cartella madre
    - pulire st_analysis.py per prendere FA e GEOM_SHAPE come paramemtri da param file, 
    --- salvare (dentro R?) per ogni p -> l'intensità media e la FA -> così poi posso filtrare la FA per intensità
    --- la media delle FA dei cubetti analizzati 
    - eseguire o no il plot_dei_quiver in base a un param in ingresso come in PIG_analysis (prendo da pig_analysis?)

    '''




