"""
    Training config classes, handles hidden layers and hyperparemters
    Inputs and outputs dynamically scale based on X and Y data structures
"""

import util
import datetime
import os
import json

class bracketClassifierConfig(object):
    """Neural network training config to hold hyperparemters"""
    def __init__(self):
        self.SEED = 42
        self.LR = 0.0005
        self.BATCH = 128
        self.EPOCHS = 100
        self.DEVICE = util.check_gpu()
        self.HIDDEN_SIZES_SHARED = [64, 256, 256, 512, 512, 512] 
        self.HIDDEN_SIZES_A = [512, 512, 256, 128, 64, 16] 
        self.HIDDEN_SIZES_B = [512, 512, 256, 128, 64, 16] 
        self.WEIGHT_DECAY = 1e-4
        self.L1_LAMBDA = 0.001 # L1 reg not used
        self.DROPOUT_RATE = 0.1
        self.USE_NOISE = False
        self.DIRICHLET_ALPHA = 0.05
        self.DIRICHLET_RATIO = 0.15
        self.WEIGHTING_POWER_A = .65
        self.WEIGHTING_POWER_B = .75
        
        with open('secrets.json', 'r') as file:
            secrets = json.load(file)
            self.BASE_PATH = secrets['models']['multioutput_base_path']
        file.close()
        
        self.CHECKPOINT = f'{self.BASE_PATH}{datetime.datetime.now().strftime("%H%M%S_%f")}.checkpoint'
        self.RELOAD = False  
        self.TRAIN = False
        self.TEST = False
        if not os.path.exists(self.BASE_PATH):
            os.makedirs(self.BASE_PATH)
            
class modalityClassifierConfig(object):
    """docstring for modalityClassifierConfig."""
    def __init__(self):
        self.DEVICE = util.check_gpu()
        self.SEED = 42
        self.LR = 0.000005
        self.BATCH = 128
        self.EPOCHS = 50
        self.HIDDEN_SIZE = 256
        self.NUM_HIDDEN = 4
        self.WEIGHT_DECAY = 1e-4
        self.L1_LAMBDA = 0.01
        self.DROPOUT_RATE = 0.0
        self.USE_NOISE = False
        self.DIRICHLET_ALPHA = 0.15
        self.DIRICHLET_RATIO = 0.15
        self.WEIGHTING_POWER = .9
        
        with open('secrets.json', 'r') as file:
            secrets = json.load(file)
            self.BASE_PATH = secrets['models']['modality_base_path']
        file.close()
        
        self.CHECKPOINT = f'{self.BASE_PATH}{datetime.datetime.now().strftime("%H%M%S_%f")}.checkpoint'
        self.RELOAD = False  
        self.TRAIN = True
        self.TEST = True
        if not os.path.exists(self.BASE_PATH):
            os.makedirs(self.BASE_PATH)