import torch
import numpy as np
import os
import argparse
from unet import *
from omegaconf import OmegaConf
from train import trainer
from feature_extractor import * 
from ddad import *
from datetime import datetime


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

def build_model(config):
    if config.model.DDADS:
        unet = UNetModel(config.data.image_size, 32, dropout=0.3, n_heads=2 ,in_channels=config.data.input_channel)
    else:
        unet = UNetModel(config.data.image_size, 64, dropout=0.0, n_heads=4 ,in_channels=config.data.input_channel)
    return unet

def train(config, category):
    torch.manual_seed(42)
    np.random.seed(42)
    unet = build_model(config)

    result_save_dir = os.path.join(os.getcwd(), config.metrics.result_dir, category)
    print(" Num params: ", sum(p.numel() for p in unet.parameters()))
    with open(os.path.join(result_save_dir, 'result.txt'), 'a', encoding='utf-8') as file:
        file.write(f'Train:\n')
        file.write(f'| Num params: {sum(p.numel() for p in unet.parameters())}\n')

    unet = unet.to(config.model.device)
    unet.train()
    unet = torch.nn.DataParallel(unet)
    # checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), category,'1000'))
    # unet.load_state_dict(checkpoint)  
    trainer(unet, category, config)#category, 


def detection(config, category):
    unet = build_model(config)
    checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, category, str(config.model.load_chp)))
    unet = torch.nn.DataParallel(unet)
    unet.load_state_dict(checkpoint)    
    unet.to(config.model.device)
    checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, category, str(config.model.load_chp)))
    unet.eval()
    ddad = DDAD(unet, category, config)
    ddad()
    

def finetuning(config, category):
    unet = build_model(config)
    checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, category, str(config.model.load_chp)))
    unet = torch.nn.DataParallel(unet)
    unet.load_state_dict(checkpoint)    
    unet.to(config.model.device)
    unet.eval()
    domain_adaptation(unet, category, config, fine_tune=True)





def parse_args():
    cmdline_parser = argparse.ArgumentParser('DDAD')    
    cmdline_parser.add_argument('-cfg', '--config', 
                                default= os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), 
                                help='config file')
    cmdline_parser.add_argument('--train', 
                                default= False, 
                                help='Train the diffusion model')
    cmdline_parser.add_argument('--detection', 
                                default= False, 
                                help='Detection anomalies')
    cmdline_parser.add_argument('--domain_adaptation', 
                                default= False, 
                                help='Domain adaptation')
    args, unknowns = cmdline_parser.parse_known_args()
    return args


    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    config = OmegaConf.load(args.config)

    print('data informations')
    print('========================')
    print(f'name: {config.data.name}')
    print(f'categories: {config.data.categories}')
    print()


    main_start_time = datetime.now()
    print(f'Main start time: {main_start_time}')
    print()

    for category in config.data.categories:
        category_start_time = datetime.now()

        result_save_dir = os.path.join(os.getcwd(), config.metrics.result_dir, category)
        if not os.path.exists('results'):
                os.mkdir('results')
        if not os.path.exists(config.metrics.result_dir):
            os.mkdir(config.metrics.result_dir)
        if not os.path.exists(result_save_dir):
            os.mkdir(result_save_dir)

        print("Class: ",category, "   w:", config.model.w, "   v:", config.model.v, "   load_chp:", config.model.load_chp,   "   feature extractor:", config.model.feature_extractor,"         w_DA: ",config.model.w_DA,"         DLlambda: ",config.model.DLlambda)
        print(f'{config.model.test_trajectoy_steps=} , {config.data.test_batch_size=}')
        print(f'Category start time: {category_start_time}')
        with open(os.path.join(result_save_dir, 'result.txt'), 'a', encoding='utf-8') as file:
            file.write(f'Class: {category}\tw: {config.model.w}\tv: {config.model.v}\tload_chp: {config.model.load_chp}\tfeature extractor: {config.model.feature_extractor}\tw_DA: {config.model.w_DA}\tDLlambda: {config.model.DLlambda}\n')
            file.write(f'{config.model.test_trajectoy_steps=}\t{config.data.test_batch_size=}\n')
            file.write('\n')

        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        if args.train:
            train_start_time = datetime.now()
            print('Training...')
            print(f'Train start time (class: {category}): {train_start_time}')

            train(config, category)

            train_end_time = datetime.now()
            print(f'Train end time (class: {category}): {train_end_time}')
            print(f'Train elapsed time (class: {category}): {train_end_time - train_start_time}')
            print()
            with open(os.path.join(result_save_dir, 'result.txt'), 'a', encoding='utf-8') as file:
                file.write(f'| Total time: {train_end_time - train_start_time}\n')
                file.write(f'| Average time: {(train_end_time - train_start_time) / config.model.epochs}\n')
                file.write('\n')


        if args.domain_adaptation:
            da_start_time = datetime.now()
            print('Domain Adaptation...')
            print(f'Domain adaptation start time (class: {category}): {da_start_time}')

            finetuning(config, category)

            da_end_time = datetime.now()
            print(f'Domain adaptation end time (class: {category}): {da_end_time}')
            print(f'Domain adaptation elapsed time (class: {category}): {da_end_time - da_start_time}')
            print()

        if args.detection:
            detect_start_time = datetime.now()
            print('Detecting Anomalies...')
            print(f'Detecting anomalies start time (class: {category}): {detect_start_time}')

            detection(config, category)

            detect_end_time = datetime.now()
            print(f'Detecting anomalies end time (class: {category}): {detect_end_time}')
            print(f'Detecting anomalies elapsed time (class: {category}): {detect_end_time - detect_start_time}')
            print()

        category_end_time = datetime.now()
        print(f'Category({category}) end time: {category_end_time}')
        print(f'Category({category}) elapsed time: {category_end_time - category_start_time}')
        print()
    
    main_end_time = datetime.now()
    print(f'Main end time: {main_end_time}')
    print(f'Main elapsed time: {main_end_time - main_start_time}')


        