'''Train PS-KD: learning with PyTorch.'''
from __future__ import print_function
import configparser
from matplotlib.contour import ContourSet

#----------------------------------------------------
#  Pytorch
#----------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import seaborn as sns
import matplotlib.pyplot as plt

#--------------
#  Datalodader
#--------------
from loader import custom_dataloader

#----------------------------------------------------
#  Load CNN-architecture
#----------------------------------------------------
from models.network import get_network

#--------------
# Util
#--------------
from utils.dir_maker import DirectroyMaker
from utils.AverageMeter import AverageMeter
from utils.metric import metric_ece_aurc_eaurc
from utils.color import Colorer
from utils.etc import progress_bar, is_main_process, save_on_master, paser_config_save, set_logging_defaults

#----------------------------------------------------
#  Etc
#----------------------------------------------------
import os, logging
import argparse
import numpy as np
import json

#----------------------------------------------------
#  Training Setting parser
#----------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='visualizer for class prototype vectors')
    parser.add_argument('--experiment_dir', type=str, default='expts',help='Directory name where the model ckpts are stored')
    args = parser.parse_args()
    return parser,args

#----------------------------------------------------
#  Colour print 
#----------------------------------------------------
C = Colorer.instance()


#----------------------------------------------------
# returns the superclass and how many indexes before it were in the same class
#----------------------------------------------------

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    j=0
    for i in range(targets+1):
       
        if coarse_labels[i]== coarse_labels[targets]:
            j= j+1
            
    return coarse_labels[targets], j-1

def main():
    parser, args = parse_args()
    config_file = os.path.join(args.experiment_dir, 'C:/Users/haziq/Desktop/DL Project/ProjectFinal/PS-KD-Pytorch/exptsnew/cifar100_SmallResNet__2022-7-9-16-47-42/config/config.json')
    model_dir = os.path.join(args.experiment_dir, 'C:/Users/haziq/Desktop/DL Project/ProjectFinal/PS-KD-Pytorch/exptsnew/cifar100_SmallResNet__2022-7-9-16-47-42/model')
    assert os.path.exists(config_file), "config file path incorrect"
    assert os.path.exists(model_dir), "model directory path incorrect"
    
    config_dict = json.load(open(config_file,'r'))
    t_args = argparse.Namespace()
    t_args.__dict__.update(config_dict)
    config_args = parser.parse_args(namespace=t_args)
    config_args.gpu = 0
    net = get_network(config_args)
    net = net.cuda()
    start_epoch = config_args.start_epoch
    end_epoch = config_args.end_epoch
    saveckp_freq = config_args.saveckp_freq
    
    sim_matrices = []
    
    for epoch in range(start_epoch, end_epoch):
        if (epoch+1) % saveckp_freq !=0 :
            continue
        checkpoint = torch.load(os.path.join(model_dir,'checkpoint_'+str(epoch)+'.pth'))  
        net.load_state_dict(checkpoint['net'])
        learnable_params = net.learnable_params.weight.data # tensor of shape = [num_classes, 512]
        learnable_params = learnable_params.clone().detach().cpu().numpy() # nparray of shape = [num_classes, 512]
        print(learnable_params.shape)
        similarity_matrix = learnable_params @ learnable_params.T
        sim_matrices.append(similarity_matrix)
        
        #print(similarity_matrix)
    
    
    sim_matrices = np.array(sim_matrices)
    np.save(open(os.path.join(model_dir,'learnable_parameters_similarity.npy'),'wb'),sim_matrices)
   
    print(len(sim_matrices))
    
    coarse_sim_matrices = np.zeros((100,100))

   
    print(coarse_sim_matrices.shape)
    
    def heatMap(sim_matrices):
        #for i in range(len(sim_matrices)):
            for x in range(100):
                for j in range(100):
                    coarse_sim_matrices[sparse2coarse(x)[0]*5 + sparse2coarse(x)[1]][sparse2coarse(j)[0]*5 + sparse2coarse(j)[1]]= sim_matrices[2][x][j]
            #coarse_sim_matrices.append(coarse_sim_matrices)
            ax = sns.heatmap(coarse_sim_matrices)
            plt.show()
            return
      
    heatMap(sim_matrices)
    
    # ax = sns.heatmap(sim_matrices[2])
    # plt.show()

if __name__ == '__main__':
    main()
