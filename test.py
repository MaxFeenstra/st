import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

from dataio.dataset_processing import DatasetInput
from model.model import Network
from utils.utils import *

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

def main(config):

    config_fold = config.config_file + str(config.fold_id) + '.json'
    json_opts = json_file_to_pyobj(config_fold)

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create experiment directories
    make_new = False
    timestamp = get_experiment_id(make_new, json_opts.experiment_dirs.load_dir, config.fold_id)
    experiment_path = 'experiments' + '/' + timestamp
    model_dir = experiment_path + '/' + json_opts.experiment_dirs.model_dir
    
    if config.test_mode == 'tcga':
        test_output_dir = experiment_path + '/' + json_opts.experiment_dirs.test_output_dir + '_tcga'
    else:
        test_output_dir = experiment_path + '/' + json_opts.experiment_dirs.test_output_dir
    make_dir(test_output_dir)

    fold_mean = json_opts.data_params.fold_means
    fold_std = json_opts.data_params.fold_stds
    assert(len(fold_mean) == len(fold_std))

    # Set up the model
    logging.info("Initialising model")
    model_opts = json_opts.model_params
    n_out_features = json_opts.data_params.n_genes

    model = Network(model_opts, n_out_features)
    model = model.to(device)

    # Dataloader
    logging.info("Preparing data")
    num_workers = json_opts.data_params.num_workers
    test_dataset = DatasetInput(json_opts.data_source, config.fold_id, fold_mean, fold_std,
                                json_opts.data_params.in_h, json_opts.data_params.in_w,
                                mode=config.test_mode, tcga_path=config.tcga_path)
    test_loader = DataLoader(dataset=test_dataset, 
                              batch_size=1, 
                              shuffle=False, num_workers=num_workers)

    n_test_examples = len(test_loader)
    logging.info("Total number of testing examples: %d" %n_test_examples)

    # Get list of model files
    if config.test_epoch < 0:
        saved_model_paths, _ = get_files_list(model_dir, ['.pth'])
        saved_model_paths = sorted_alphanumeric(saved_model_paths)
        saved_model_epochs = [(os.path.basename(x)).split('.')[0] for x in saved_model_paths]
        saved_model_epochs = [x.split('_')[-1] for x in saved_model_epochs]
        if config.test_epoch == -2:
            saved_model_epochs = np.array(saved_model_epochs, dtype='int')
        elif config.test_epoch == -1:
            saved_model_epochs = np.array(saved_model_epochs[-1], dtype='int')
            saved_model_epochs = [saved_model_epochs]
    else:
        saved_model_epochs = [config.test_epoch]

    logging.info("Begin testing")

    mae_epochs = np.zeros((len(saved_model_epochs), n_out_features))
    mse_epochs = np.zeros((len(saved_model_epochs), n_out_features))
    pcc_epochs = np.zeros((len(saved_model_epochs), n_out_features))
    mae_epochs_avg = np.zeros(len(saved_model_epochs))
    mse_epochs_avg = np.zeros(len(saved_model_epochs))
    pcc_epochs_avg = np.zeros(len(saved_model_epochs))

    if config.test_mode == 'tcga':

        for epoch_idx, test_epoch in enumerate(saved_model_epochs):

            pred_all = np.zeros((n_test_examples, n_out_features))

            # Restore model
            load_path = model_dir + "/epoch_%d.pth" %(test_epoch)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            assert(epoch == test_epoch)
            print("Testing " + load_path)

            model = model.eval()

            # Write predictions to text file
            txt_path = test_output_dir + '/' + 'epoch_' + str(test_epoch) + '.txt'
            
            with open(txt_path, 'w') as output_file:

                for batch_idx, (batch_x, ID) in enumerate(test_loader):

                    # Transfer to GPU
                    batch_x = batch_x.to(device)

                    # Forward pass
                    y_pred = model(batch_x)

                    # Labels, predictions per example
                    pred_all[batch_idx,:] = y_pred.squeeze().detach().cpu().numpy()

                    for f in range(n_out_features):
                        output_file.write(ID[0] + ' ' + str(f) + ' predict: ' + str(pred_all[batch_idx,f]) + '\n')
    
    else:

        for epoch_idx, test_epoch in enumerate(saved_model_epochs):

            gt_all = np.zeros((n_test_examples, n_out_features))
            pred_all = np.zeros((n_test_examples, n_out_features))

            # Restore model
            load_path = model_dir + "/epoch_%d.pth" %(test_epoch)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            assert(epoch == test_epoch)
            print("Testing " + load_path)

            model = model.eval()

            # Write predictions to text file
            txt_path = test_output_dir + '/' + 'epoch_' + str(test_epoch) + '.txt'
            
            with open(txt_path, 'w') as output_file:

                for batch_idx, (batch_x, batch_y, ID) in enumerate(test_loader):

                    # Transfer to GPU
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    # Forward pass
                    y_pred = model(batch_x)

                    # Labels, predictions per example
                    gt_all[batch_idx,:] = batch_y.squeeze().detach().cpu().numpy()
                    pred_all[batch_idx,:] = y_pred.squeeze().detach().cpu().numpy()

                    for f in range(n_out_features):
                        output_file.write(ID[0] + ' ' + str(f) + ' gt: ' + str(gt_all[batch_idx,f]) + ' predict: ' + str(pred_all[batch_idx,f]) + '\n')

                # Compute performance
                mae_all = np.zeros(n_out_features)
                mse_all = np.zeros(n_out_features)
                pcc_all = np.zeros(n_out_features)

                for f in range(n_out_features):
                    mae_all[f] = mean_absolute_error(gt_all[:,f], pred_all[:,f])
                    mse_all[f] = mean_squared_error(gt_all[:,f], pred_all[:,f])
                    pcc_all[f] = stats.pearsonr(gt_all[:,f], pred_all[:,f])[0]

                output_file.write('Overall MAE, MSE, and PCC \n')
                output_file.write(" ".join(map(str, np.around(mae_all, 5))))
                output_file.write('\n')
                output_file.write(" ".join(map(str, np.around(mse_all, 5))))
                output_file.write('\n')
                output_file.write(" ".join(map(str, np.around(pcc_all, 5))))
                output_file.write('\n')
                output_file.write(str(np.around(np.mean(mae_all), 5)))
                output_file.write('\n')
                output_file.write(str(np.around(np.mean(mse_all), 5)))
                output_file.write('\n')
                output_file.write(str(np.around(np.mean(pcc_all), 5)))

            # Store performance for each feature
            mae_epochs[epoch_idx,:] = mae_all[:]
            mse_epochs[epoch_idx,:] = mse_all[:]
            pcc_epochs[epoch_idx,:] = pcc_all[:]

            # Means for this epoch
            mae_epochs_avg[epoch_idx] = np.mean(mae_all)
            mse_epochs_avg[epoch_idx] = np.mean(mse_all)
            pcc_epochs_avg[epoch_idx] = np.mean(pcc_all)
            print('MAE mean: ', mae_epochs_avg[epoch_idx])
            print('MSE mean: ', mse_epochs_avg[epoch_idx])
            print('PCC mean: ', pcc_epochs_avg[epoch_idx])

        best_epoch = np.argmax(pcc_epochs_avg)
        print('Best pearson: epoch %d, coef %.7f' %(saved_model_epochs[best_epoch], pcc_epochs_avg[best_epoch]))
        print('Best mae %.7f, mse %.7f' %(mae_epochs_avg[best_epoch], mse_epochs_avg[best_epoch]))

    logging.info("Testing finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', default='configs/config', type=str,
                        help='config file path')
    parser.add_argument('--test_epoch', default=-2, type=int,
                        help='test model from this epoch, -1 for last, -2 for all')
    parser.add_argument('--fold_id', default=1, type=int,
                        help='which cross-validation fold')
    parser.add_argument('--test_mode', default='tcga', type=str)
    parser.add_argument('--tcga_path', default='/mnt/HDD4/Helen/Benchmarking_Histology/tcga-brca_processed/patches_224', type=str)
    
    config = parser.parse_args()
    main(config)
