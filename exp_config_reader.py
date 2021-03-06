from argments import args
import os
import os.path as osp
import yaml
import getpass


def get_config(config_path):
    assert osp.isfile(config_path), "the exp_run_config yaml file doesn't exist: {}".format(config_path)
    with open(str(config_path), 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded


"""
Configs
"""
PROJECT_NAME = "video_recognition"
userName = getpass.getuser()
ROOT = osp.join("/home", userName)

config_file = args.exp_run_config
config_path = osp.join('exp_config', config_file)
config = get_config(config_path)

## get env var from configs
# Experiment info
EXP_NAME = config['exp_name']
USE_TEST = config['use_test']
N_TEST_INTERVAL = config['n_test_interval']
IF_PRETRAIN = config['if_pretrain']

# Load saved model exp_run_config
RESUM_MODEL_PATH = config['resume_model_path']
RESUM_EPOCH = config['resume_epoch']
try:
    RESUM_OPTIMIZER = config['resume_optimizer']
except:
    RESUM_OPTIMIZER = False

# Dataset
DATA_PATH = config['data_path']
DATASET = config['dataset']
N_WORKERS = config['n_workers']

# Log & Save
# log path
LOG_ROOT = osp.join("exp_logs", EXP_NAME)
LOG_PATH = osp.join(LOG_ROOT, EXP_NAME)
# save path
SAVE_ROOT = osp.join("saved_models", EXP_NAME)

# Model exp_run_config
MODEL_NAME = config['model_name']
BATCH_SIZE = config['batch_size']
SNAPSHOT = config['snapshot']
MAX_EPOCH = config['max_epoch']
N_CLASSES = config['n_classes']
CLIP_LEN = config['clip_len']
GRAY_SCALE = config['gray_scale']
OPTIMIZER = config['optimizer']
# aug for optimizers:
INIT_LEARNING_RATE = config['init_learning_rate']
SCHEDULER_STEP_SIZE = config['scheduler_step_size']
SCHEDULER_GAMMA = config['scheduler_gamma']
MOMENTUM = config['momentum']
WD = config['wd']