from asyncio import constants
from typing import Any
import torch
from unet import *
from dataset import *
from visualize import *
from anomaly_map import *
from metrics import *
from feature_extractor import *
from reconstruction import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

class DDAD:
    def __init__(self, unet, category, config) -> None:
        self.test_dataset = Dataset_maker(
            root= config.data.data_dir,
            #category=config.data.category,
            category = category,
            config = config,
            is_train=False,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size= config.data.test_batch_size,
            shuffle=False,
            num_workers= config.model.num_workers,
            drop_last=False,
        )
        self.unet = unet
        self.category = category
        self.config = config
        self.reconstruction = Reconstruction(self.unet, self.config)
        self.transform = transforms.Compose([
                            transforms.CenterCrop((224)), 
                        ])

    def __call__(self) -> Any:
        #feature_extractor = domain_adaptation(self.unet, self.config, fine_tune=False)
        feature_extractor = domain_adaptation(self.unet, self.category, self.config, fine_tune=False)
        feature_extractor.eval()
        
        labels_list = []
        predictions= []
        anomaly_map_list = []
        gt_list = []
        reconstructed_list = []
        forward_list = []



        with torch.no_grad():
            for input, gt, labels in self.testloader:
                input = input.to(self.config.model.device)
                x0 = self.reconstruction(input, input, self.config.model.w)[-1]
                anomaly_map = heat_map(x0, input, feature_extractor, self.config)

                anomaly_map = self.transform(anomaly_map)
                gt = self.transform(gt)

                forward_list.append(input)
                anomaly_map_list.append(anomaly_map)


                gt_list.append(gt)
                reconstructed_list.append(x0)
                for pred, label in zip(anomaly_map, labels):
                    labels_list.append(0 if label == 'good' else 1)
                    predictions.append(torch.max(pred).item())

        
        result_save_dir = os.path.join(os.getcwd(), self.config.metrics.result_dir, self.category)
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists(self.config.metrics.result_dir):
            os.mkdir(self.config.metrics.result_dir)
        if not os.path.exists(result_save_dir):
            os.mkdir(result_save_dir)
        
        with open(os.path.join(result_save_dir, 'result.txt'), 'a', encoding='utf-8') as file:
            file.write('Detection:\n')

        metric = Metric(labels_list, predictions, anomaly_map_list, gt_list, self.category, self.config)
        metric.optimal_threshold()
        if self.config.metrics.auroc:
            print('AUROC: ({:.1f},{:.1f})'.format(metric.image_auroc() * 100, metric.pixel_auroc() * 100))
            with open(os.path.join(result_save_dir, 'result.txt'), 'a', encoding='utf-8') as file:
                file.write('| AUROC: ({:.1f},{:.1f})\n'.format(metric.image_auroc() * 100, metric.pixel_auroc() * 100))

        if self.config.metrics.pro:
            print('PRO: {:.1f}'.format(metric.pixel_pro() * 100))
            with open(os.path.join(result_save_dir, 'result.txt'), 'a', encoding='utf-8') as file:
                file.write('| PRO: {:.1f}\n'.format(metric.pixel_pro() * 100))

        if self.config.metrics.misclassifications:
            metric.miscalssified()
        reconstructed_list = torch.cat(reconstructed_list, dim=0)
        forward_list = torch.cat(forward_list, dim=0)
        anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
        pred_mask = (anomaly_map_list > metric.threshold).float()
        gt_list = torch.cat(gt_list, dim=0)
        if self.config.metrics.visualisation:
            #visualize(forward_list, reconstructed_list, gt_list, pred_mask, anomaly_map_list, self.config.data.category)
            visualize(forward_list, reconstructed_list, gt_list, pred_mask, anomaly_map_list, result_save_dir, self.category)
