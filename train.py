import torch
import os
import torch.nn as nn
from dataset import *

from dataset import *
from loss import *

from datetime import datetime


def trainer(model, category, config):
    '''
    Training the UNet model
    :param model: the UNet model
    :param category: the category of the dataset
    '''
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=config.model.learning_rate)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay
    )
    train_dataset = Dataset_maker(
        root= config.data.data_dir,
        category=category,
        config = config,
        is_train=True,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
    )
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(config.model.checkpoint_dir):
        os.mkdir(config.model.checkpoint_dir)


    '''if config.model.save_model:
        model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        
        file_list = os.listdir(model_save_dir)


        torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch+1)))'''

    epoch_start_times = []
    epoch_end_times = []
    for epoch in range(config.model.epochs):
        epoch_start_times.append(datetime.now())
        for step, batch in enumerate(trainloader):
            optimizer.zero_grad()
            # loss = 0
            # for _ in range(2):
            t = torch.randint(0, config.model.trajectory_steps, (batch[0].shape[0],), device=config.model.device).long()
            loss = get_loss(model, batch[0], t, config) 
            loss.backward()
            optimizer.step()

        epoch_end_times.append(datetime.now())

        if (epoch+1) % config.model.progress_display_interval == 0:
            print(f"Epoch {epoch+1:4d} | Loss: {loss.item()}\tTotal time: {epoch_end_times[-1] - epoch_start_times[-25]}\tAverage time: {(epoch_end_times[-1] - epoch_start_times[-25])/25}")
        if (epoch+1) % config.model.save_model_interval == 0:
            if config.model.save_model:
                model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)
                torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch+1)))
                
    print(f'Category: {category}\t| Total time: {epoch_end_times[-1]-epoch_start_times[0]} \tAverage time: {(epoch_end_times[-1]-epoch_start_times[0])/config.model.epochs}')
    print()