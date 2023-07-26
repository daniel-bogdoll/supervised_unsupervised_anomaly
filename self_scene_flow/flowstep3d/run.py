import os
import yaml
import argparse
import datetime
import os.path as osp
import torch
from tqdm import tqdm

from models import *
from pytorch_lightning import Trainer
from experiment import SceneFlowExp
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import lightning_lite as pl


def main(config):

    # Create Neptune Logger
    neptune_logger = NeptuneLogger(
                                   api_key=None,
                                   offline_mode=config['logging_params']['offline_mode'],
                                   project_name=config['logging_params']['project_name'],
                                   experiment_name=config['logging_params']['exp_name'],
                                   params={**config['exp_params'], **config['model_params'], **config['trainer_params']},
                                   tags=config['logging_params']['tags'])

    # Create model and experiment instance
    model = models_dict[config['model_params']['model_name']](**config['model_params'])
    experiment = SceneFlowExp(model, config['exp_params'])

    if 'pre_trained_weights_checkpoint' in config['exp_params'].keys():
        print(f"Loading pre-trained model: {config['exp_params']['pre_trained_weights_checkpoint']}")
        checkpoint = torch.load(config['exp_params']['pre_trained_weights_checkpoint'], map_location=lambda storage, loc: storage)
        experiment.load_state_dict(checkpoint['state_dict'])

    # Create a trainer instance
    # use trainer_params to set num_nodes and gpus
    if config['train']:
        time_str = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        exp_ckpt_dir = osp.join(config['logging_params']['ckpt_dir'], time_str)
    else:
        exp_ckpt_dir = osp.join(config['logging_params']['ckpt_dir'], 'test')
    os.makedirs(exp_ckpt_dir, exist_ok=True)
    ckpt_callback = ModelCheckpoint(dirpath=osp.join(exp_ckpt_dir, '{epoch}'),
                                    save_last=True
                                    )
    trainer = Trainer(logger=neptune_logger,
                      callbacks=[ckpt_callback],
                      **config['trainer_params'])

    #trainer.ckpt_path = 'checkpoints/flowstep3d_self/2021-03-24_01-03/epoch=23.ckpt'
    if config['train']:
        print('Start Training!')
        trainer.fit(experiment)
    else:
        print('Start Testing')
        trainer.test(experiment)


#def start_flowstep3d(config_path):
if __name__ == '__main__':
    
    # Load config file from input arguments
    parser = argparse.ArgumentParser(description='Generic runner for Scene-Flow models')
    parser.add_argument('--config', '-c',
                        dest='filename',
                        metavar='FILE',
                        help='Path to .yaml config file for the experiment',
                        default='configs/test/flowstep3d_self.yaml')
                    
    args = parser.parse_args()
    
    with open(args.filename, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    pl.utilities.seed.seed_everything(seed=18)
    # adjusted for more than one sequence
    sequences = config['exp_params']['data']['sequence']
    for seq in tqdm(sequences):
        config['exp_params']['data']['sequence'] = seq
        
        # Run
        main(config)
    
