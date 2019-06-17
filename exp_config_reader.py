from argments import args
import os.path
import yaml


def get_config(config_path):
    assert os.path.isfile(config_path), "the exp_run_config yaml file doesn't exist: {}".format(config_path)
    with open(str(config_path), 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded


config_file = args.exp_run_config
config_path = os.path.join('exp_config', config_file)
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

# Dataset
DATA_PATH = config['data_path']
DATASET = config['dataset']
N_WORKERS = config['n_workers']

# Model exp_run_config
MODEL_NAME = config['model_name']
BATCH_SIZE = config['batch_size']
SNAPSHOT = config['snapshot']
MAX_EPOCH = config['max_epoch']
# aug for optimizers:
INIT_LEARNING_RATE = config['init_learning_rate']
MOMENTUM = config['momentum']
WD = config['wd']