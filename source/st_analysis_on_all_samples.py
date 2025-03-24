# run st_analysis.py on all the samples in the input basepath
# example:
# .../basepath/N1/ <- {N1.tif, parameters.txt}
# .../basepath/N2/ <- {N2.tif, parameters.txt}
# .../basepath/N2/ <- {N3.tif, parameters.txt}
# > python disarray_on_all -s /.../basepath   -> run st_analysis.py -s /.../N1; st_analysis.py -s /.../N1 etc. 

import argparse
import os
import subprocess

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
    source_path        = args.source_path[0]
    _disarray_analysis = args.disarray
    _skip_st_analysis  = args.skip_st_analysis

    # list of subfolders
    samples = os.listdir(source_path)
    samples.sort()

    print(Bcolors.WARNING + '\n\n**************** Start Fibers Analysis on all Samples in :' + Bcolors.ENDC)
    print('- source_path : ', source_path)
    print('- list of samples:')
    for s in samples: 
        print('   -', s)
    print('- [optional] Perform Disarray analysis: ', _disarray_analysis)
    print('- [optional] Skip Structure Tensor Analysis: ', _skip_st_analysis)
    print()

    # run analisys on each sample
    for s in samples:
        print(Bcolors.WARNING + '\n Start the analysis on: Sample {}\n'.format(s) + Bcolors.ENDC)
        
        # sample path
        spath = os.path.join(source_path, s)

        # parameters file on the current path
        par_fnames = [p for p in os.listdir(spath) if p.startswith('parameters')]

        if not _skip_st_analysis:

            # tiff file here
            tiff_fnames = [t for t in os.listdir(spath) if t.endswith('.tif') or t.endswith('.tiff')]
            tiffpath = os.path.join(spath, tiff_fnames[0])

            # check if there is only one sample and one param file in the current directory
            if (len(tiff_fnames), len(par_fnames)) == (1, 1):

                # perform analsys calling the sub-script
                os.system('python st_analysis.py -s {} -p {}'.format(tiffpath, par_fnames[0]))
                # subprocess.call(['secondary.py', '-sf', mask_path])

        if _disarray_analysis is True:

            # search file numpy R in the current path
            R_fnames = [r for r in os.listdir(spath) if r.startswith('R_') and r.endswith('.npy')]

            # check if there is only one R
            if len(R_fnames) == 1:
                R_path = os.path.join(spath, R_fnames[0])

                # perform disarray analsys calling the sub-script
                os.system('python local_disarray_by_R.py -r {} -p {}'.format(R_path, par_fnames[0]))
                # subprocess.call(['secondary.py', '-sf', mask_path])

    print(Bcolors.WARNING + '\n Finish to analyze all samples\n' + Bcolors.ENDC)
    return None

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run st_analysys on all samples in the input path')
    parser.add_argument('-s', '--source_path', nargs='+', required=False,
                        help='Perform st_analysis on all samples in input path.')
    parser.add_argument('-d', '--disarray', action='store_true', default=True, required=False,
                        help='if passed, perform disarray analysis on all samples after the st_analysis.py')
    parser.add_argument('-x', '--skip-st-analysis', action='store_true', default=False, required=False,
                        help='if passed, the script skip the st_analysis.')

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




