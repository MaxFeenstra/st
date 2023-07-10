import os 
from fnmatch import fnmatch
import re
import argparse
import numpy as np
import shutil 
from tqdm import tqdm

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

    test_output_dir = experiments_dir + '/' + timestamp + '/' + config.test_output_dir
    test_output_files, _ = get_files_list(test_output_dir, ext_array=['.txt'])
    test_output_files = sorted_alphanumeric(test_output_files)

    # Determine the best performing model 
    mean_pcc_all = np.zeros(len(test_output_files))
    
    for i, output_file in enumerate(tqdm(test_output_files)):
        with open(output_file, 'r') as f:
            file_lines = f.readlines()
            # mean_pcc_all[i] = float(file_lines[-1])
            pcc_line = file_lines[-5]
            pcc_line_array = np.asarray([float(s) for s in pcc_line.split(' ')])
            mean_pcc_all[i] = np.mean(np.abs(pcc_line_array))
            
    best_idx = np.argmax(mean_pcc_all)
    best_epoch = os.path.basename(test_output_files[best_idx]).split('.')[0] + '.pth'
    model_keep = experiments_dir + '/' + timestamp + '/' + config.model_dir + '/' + best_epoch
    # best_idx2 = np.argmax(mean_f1_all)
    # best_epoch2 = os.path.basename(test_output_files[best_idx2]).split('.')[0] + '.pth'
    # model_keep2 = experiments_dir + '/' + timestamp + '/' + config.model_dir + '/' + best_epoch2
    
    models_del_list, _ = get_files_list(experiments_dir + '/' + timestamp + '/' + config.model_dir, ['.pth'])
    models_del_list.remove(model_keep)
    # if model_keep != model_keep2:
        # models_del_list.remove(model_keep2)
    assert(models_del_list.count(model_keep) == 0)
    # assert(models_del_list.count(model_keep2) == 0)
    
    print(model_keep, mean_pcc_all[best_idx])
    # for path in models_del_list:
        # # print(path)
        # os.remove(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--network_dir', type=str, default='', help='base directory of model')
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--load_dir', type=str, default='last', help='name of the folder to load model from')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--test_output_dir', type=str, default='val_output')

    config = parser.parse_args()
    main(config)
