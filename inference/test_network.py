
from __future__ import print_function
import sys

sys.path.append('./auxiliary/')
sys.path.append('./')
import numpy as np
import torch.optim as optim
import torch.nn as nn
import model
import ply
import time
from sklearn.neighbors import NearestNeighbors

sys.path.append("./extension/")
sys.path.append("./training/")
sys.path.append("/app/python/")
import dist_chamfer as ext

distChamfer = ext.chamferDist()
import trimesh
import torch
import pandas as pd
import os
from mpl_toolkits import mplot3d
import argument_parser
import my_utils
import trainer
import matplotlib.pyplot as plt
from termcolor import colored
import pointcloud_processor

class Inference(object):
    def __init__(self, model_path="", num_points=6890, save_path=None, network=None):
        self.num_points = num_points
        self.model_path = model_path
        self.save_path = save_path
        
        # load network
        if network is None:
            self.network = model.OEMDNet(num_points=self.num_points)
            self.network.cuda()
            self.network.apply(my_utils.weights_init)
            if self.model_path != '':
                print("Reload weights from: ", self.model_path)
                self.network.load_state_dict(torch.load(self.model_path))
        else:
            self.network = network
        self.network.eval()
        
    def reconstruct(self, input_p):
        print("Reconstructing ", input_p)
        input = trimesh.load(input_p, process=False)
        
    def run(self, input, scalefactor, path):
        