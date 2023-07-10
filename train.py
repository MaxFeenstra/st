import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from dataio.dataset_processing import DatasetInput
from model.model import Network
from utils.utils import *

import numpy as np
from tqdm import tqdm

def main(config):

    config_fold = config.config_file + str(config.fold_id) + '.json'
    json_opts = json_file_to_pyobj(config_fold)
    #print(json_opts)
    #json_opts.data_source = './her2st_processing/patches_resized'
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    #print('working')
    # Create experiment directories
    if config.resume_epoch == None:
        make_new = True 
    else:
        make_new = False
    timestamp = get_experiment_id(make_new, json_opts.experiment_dirs.load_dir, config.fold_id)
    experiment_path = 'experiments' + '/' + timestamp
    make_dir(experiment_path + '/' + json_opts.experiment_dirs.model_dir)

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
    train_dataset = DatasetInput(json_opts.data_source, config.fold_id, fold_mean, fold_std,
                                json_opts.data_params.in_h, json_opts.data_params.in_w,
                                mode='train')
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=json_opts.training_params.batch_size, 
                              shuffle=True, num_workers=num_workers, drop_last=True)
    #print('here')
    n_train_examples = len(train_loader)
    logging.info("Total number of training examples: %d" %n_train_examples)

    # Auxiliary losses and optimiser
    criterion_mae = torch.nn.MSELoss(reduction='sum')
    # optimizer = torch.optim.Adam(model.parameters(), lr=json_opts.training_params.learning_rate, 
    #                              betas=(json_opts.training_params.beta1, 
    #                                     json_opts.training_params.beta2),
    #                              weight_decay=json_opts.training_params.l2_reg_alpha)
    optimizer = torch.optim.SGD(model.parameters(), lr=json_opts.training_params.learning_rate, momentum=0.9)

    if config.resume_epoch != None:
        initial_epoch = config.resume_epoch
    else:
        initial_epoch = 0

    # Restore saved model
    if config.resume_epoch != None:
        load_path = experiment_path + '/' + json_opts.experiment_dirs.model_dir + \
                    "/epoch_%d.pth" %(config.resume_epoch)
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        assert(epoch == config.resume_epoch)
        print("Resume training, successfully loaded " + load_path)
    # else:
    #     model_sd = model.state_dict()

    #     pre_trained_model = torch.load("resnet50-19c8e357.pth")
    #     pre_trained_dict = list(pre_trained_model.items())

    #     # Match keys to load weights
    #     for key, value in pre_trained_dict:
    #         # Ignore fc layers
    #         if "fc" in key:
    #             continue

    #         # "Gate" layers (first 7x7 conv)
    #         elif "layer" not in key:
    #             if "conv" in key:
    #                 key_equiv = "resnet_stage_0.encoder.gate.0.weight"
    #             else:
    #                 key_equiv = key.replace("bn1","resnet_stage_0.encoder.gate.1")

    #         # Shortcut/downsample layers
    #         elif "downsample" in key:
    #             stage_idx = int(key.split('.')[0].replace("layer","")) - 1

    #             key_equiv = key.replace("downsample", "shortcut")
    #             prefix = "resnet_stage_%d.encoder.blocks.blocks" %stage_idx
    #             key_equiv = key_equiv.replace("layer"+str(stage_idx+1), prefix)

    #         # Residual blocks
    #         else:
    #             stage_idx = int(key.split('.')[0].replace("layer","")) - 1
                
    #             prefix = "resnet_stage_%d.encoder.blocks.blocks." %stage_idx

    #             block_part = str(key.split(".")[1]) + ".blocks."
                
    #             unit_idx = int((key.split(".")[2])[-1])
    #             # 1->0, 2->2, 3->4
    #             if unit_idx == 1:
    #                 equiv_unit_idx = 0
    #             elif unit_idx == 2:
    #                 equiv_unit_idx = 2
    #             else:
    #                 equiv_unit_idx = 4

    #             if "conv" in key:
    #                 unit_type = ".0.weight"
    #             else:
    #                 unit_type = ".1." + key.split(".")[-1]

    #             key_equiv = prefix + block_part + str(equiv_unit_idx) + unit_type

    #         model_sd[key_equiv].copy_(value)


    logging.info("Begin training")

    model = model.train()

    for epoch in tqdm(range(initial_epoch, json_opts.training_params.total_epochs)):
        epoch_train_loss = 0.

        for _, (batch_x, batch_y) in enumerate(train_loader):

            # Transfer to GPU
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            final_pred = model(batch_x)

            # Optimisation
            loss = criterion_mae(final_pred, 
                                 batch_y).squeeze()

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.detach().cpu().numpy()
                
        # Save model
        if (epoch % json_opts.save_freqs.model_freq) == 0:
            save_path = experiment_path + '/' + json_opts.experiment_dirs.model_dir + \
                        "/epoch_%d.pth" %(epoch+1)
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, save_path)
            logging.info("Model saved: %s" % save_path)

        # Print training loss every epoch
        print('Epoch[{}/{}], total loss:{:.4f}'.format(epoch+1, json_opts.training_params.total_epochs, 
                                                       epoch_train_loss))

    logging.info("Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', default='configs/config', type=str,
                        help='config file path')
    parser.add_argument('--resume_epoch', default=None, type=int,
                        help='resume training from this epoch, set to None for new training')
    parser.add_argument('--fold_id', default=1, type=int,
                        help='which cross-validation fold')

    config = parser.parse_args()
    #print('W')
    main(config)
