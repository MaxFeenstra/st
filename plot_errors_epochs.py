import os 
import re
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
Alphanumerically sort a list
"""
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

"""
Get all files in a directory with a specific extension
"""
def get_files_list(path, ext_array=['.txt']):
    files_list = list()
    dirs_list = list()

    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            if any(x in file for x in ext_array):
                files_list.append(os.path.join(root, file))
                folder = os.path.dirname(os.path.join(root, file))
                # print(folder)
                if folder not in dirs_list:
                    dirs_list.append(folder)

    return files_list, dirs_list

"""
Main
"""
def main(config):

    experiments_dir = config.network_dir + 'experiments'
    print(experiments_dir)
    if config.load_dir == 'last':
        folders = next(os.walk(experiments_dir))[1]
        folders = [x for x in folders if '202' in x]
        folders = [x for x in folders if ('fold' + str(config.fold_id) + '_') in x]
        folders = sorted_alphanumeric(folders)
        folder_last = folders[-1]
        timestamp = folder_last.replace('\\','/')
    else:
        timestamp = config.load_dir

    modes = ['val', 'test']

    for i_mode, mode in enumerate(modes):

        if mode == 'val':
            line_num = -5
        else:
            line_num = -1

        test_output_dir = experiments_dir + '/' + timestamp + '/' + mode + config.output_dir
        test_output_files, _ = get_files_list(test_output_dir, ext_array=['.txt'])
        test_output_files = sorted_alphanumeric(test_output_files)

        if i_mode == 0:
            mean_all = np.zeros((len(modes), len(test_output_files)))
        
        for i, output_file in enumerate(tqdm(test_output_files)):
            with open(output_file, 'r') as f:
                file_lines = f.readlines()
                metrics_line = file_lines[line_num]
                metrics_line_array = np.asarray([float(s) for s in metrics_line.split(' ')])
                mean_all[i_mode,i] = np.mean(np.abs(metrics_line_array))
            
    plt.plot(mean_all[0,:], label = "Val MSE")
    plt.plot(mean_all[1,:], label = "Test MSE")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--network_dir', type=str, default='', help='base directory of model')
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--load_dir', type=str, default='last', help='name of the folder to load model from')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--output_dir', type=str, default='_output')

    config = parser.parse_args()
    main(config)
