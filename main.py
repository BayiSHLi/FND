import argparse
import os
import random
import warnings
import numpy as np
import torch
from run import Run
import yaml

import collections
import copy
import json
import os
import time
from datetime import datetime
import torch
from torch.utils.data import random_split
from model.models import *
from utils import *
# from moviepy.editor import AudioFileClip
from model.Pytorch_C3D_Feature_Extractor.feature_extractor_vid import c3d_feature_extractor, vgg19_extractor
import pickle
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='baseline-0930')
    parser.add_argument('--mode', default='V+T', choices=['V', 'T', 'A', 'V+T', 'V+A', 'T+A', 'V+T+A'])
    parser.add_argument('--model_name', default='FakeSV')
    parser.add_argument('--model_cfg', default='./cfg/model/FakeSV.yml')
    parser.add_argument('--data', default= 'FakeTT')
    parser.add_argument('--data_cfg', default= './cfg/data/FakeTT.yml')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--epoch_stop', type=int, default=8)
    parser.add_argument('--batch_size', type = int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--path_param', default= './checkpoints/')
    parser.add_argument('--path_tensorboard', default= './tb/')
    parser.add_argument('--shielding_prob', type=float, default= 0.3)
    parser.add_argument('--average_weight', default=True, type=bool)
    parser.add_argument('--random_repeat', default=3, type=int)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()

    print(args)
    return args


def init_exp(args, model_cfg, data_cfg):

    config = {
        'exp_name': args.exp_name,
        'mode': args.mode,
        'model_cfg': model_cfg,
        'data_root': args.data_root,
        'data': args.data,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'epoch_stop': args.epoch_stop,
        'seed': args.seed,
        'device': args.gpu,
        'lr': args.lr,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        'path_param': args.path_param,
        'path_tensorboard': args.path_tensorboard,
        'shielding_prob': args.shielding_prob,
        'average_weight': args.average_weight,
        'random_repeat': args.random_repeat,
    }

    current_date = datetime.now().strftime("%Y-%m-%d")
    save_param_path = args['path_param'] + data_cfg["data_type"] + "/" \
                           + model_cfg['model_name'] + "/" + current_date + '/'
    if not os.path.exists(save_param_path):
        os.makedirs(save_param_path)
    existing_folders = [f for f in os.listdir(save_param_path)
                        if os.path.isdir(os.path.join(save_param_path, f))]
    numbered_folders = sorted([int(f) for f in existing_folders if f.isdigit()])
    if len(numbered_folders) == 0:
        new_folder_name = "001"
    else:
        new_folder_number = numbered_folders[-1] + 1
        new_folder_name = f"{new_folder_number:03d}"
    save_param_path = save_param_path + new_folder_name + '/'
    os.makedirs(save_param_path, exist_ok=True)
    config["save_param_path"] = save_param_path

    return config


def get_data(args, data_cfg):

    dataset = FIDESDataset(args.data_root, data_cfg)
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    return train_set, test_set

def get_model(args, model_cfg):
    model = FakeSVModel(model_cfg=model_cfg,
                        dropout=args.dropout, )
    return model


def run(args, model_cfg, data_cfg):
    config = init_exp(args, model_cfg, data_cfg)
    print('-' * 50)
    print('start training')
    print('-' * 50)
    results = {"video_only": {'pred': [], 'label': [], 'mask': (1, 0, 0)},
               "text_only": {'pred': [], 'label': [], 'mask': (0, 1, 0)},
               "audio_only": {'pred': [], 'label': [], 'mask': (0, 0, 1)},
               "video_text": {'pred': [], 'label': [], 'mask': (1, 1, 0)},
               "video_audio": {'pred': [], 'label': [], 'mask': (1, 0, 1)},
               "text_audio": {'pred': [], 'label': [], 'mask': (0, 1, 1)},
               "all_modalities": {'pred': [], 'label': [], 'mask': (1, 1, 1)}}
    for repeat in range(args.random_repeat):
        train_set, test_set = get_data(args, data_cfg)

        model = get_model(args, model_cfg)

        trainer = Trainer(model=model,
                          model_cfg=model_cfg,
                          train_set=train_set,
                          test_set=test_set,
                          results=copy.deepcopy(results),
                          repeat=repeat,
                          **config
                          )
        one_repeat_result = trainer.train()
        for name in results.keys():
            results[name]['pred'] = results[name]['pred'] + one_repeat_result[name]['pred']
            results[name]['label'] = results[name]['label'] + one_repeat_result[name]['label']

    for name in results.keys():
        pred = results[name]['pred']
        label = results[name]['label']
        print(f"Results for {name}:")
        acc, report, cm = get_confusionmatrix_fnd(np.array(pred), np.array(label))

    # else:
    #     print ("Not Available")

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with open(args.model_cfg, 'r', encoding='utf-8') as f:
        model_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    with open(args.data_cfg, 'r', encoding='utf-8') as f:
        data_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    run(args, model_cfg, data_cfg)


def pad_sequence(seq_len, lst, emb):
    result = []
    for video in lst:
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len = video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len, emb], dtype=torch.long)
        elif ori_len >= seq_len:
            if emb == 200:
                video = torch.FloatTensor(video[:seq_len])
            else:
                video = torch.LongTensor(video[:seq_len])
        else:
            video = torch.cat([video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.long)], dim=0)
            if emb == 200:
                video = torch.FloatTensor(video)
            else:
                video = torch.LongTensor(video)
        result.append(video)
    return torch.stack(result)


def pad_sequence_bbox(seq_len, lst):
    result = []
    for video in lst:
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len = video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len, 45, 4096], dtype=torch.float)
        elif ori_len >= seq_len:
            video = torch.FloatTensor(video[:seq_len])
        else:
            video = torch.cat([video, torch.zeros([seq_len - ori_len, 45, 4096], dtype=torch.float)], dim=0)
        result.append(video)
    return torch.stack(result)


class Run():
    def __init__(self,
                 config
                 ):
        self.config = config
        self.model_cfg = config['model_cfg']
        self.data_type = 'FIDES'
        self.data_root = config['data_root']
        self.data = config['data']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.dropout = config['dropout']
        self.random_repeat = config['random_repeat']

        self.seed = config['seed']

        current_date = datetime.now().strftime("%Y-%m-%d")
        self.save_param_path = config['path_param'] + self.data_type + "/" \
                               + self.model_cfg['model_name'] + "/" + current_date + '/'
        if not os.path.exists(self.save_param_path):
            os.makedirs(self.save_param_path)
        existing_folders = [f for f in os.listdir(self.save_param_path)
                            if os.path.isdir(os.path.join(self.save_param_path, f))]
        numbered_folders = sorted([int(f) for f in existing_folders if f.isdigit()])
        if len(numbered_folders) == 0:
            new_folder_name = "001"
        else:
            new_folder_number = numbered_folders[-1] + 1
            new_folder_name = f"{new_folder_number:03d}"
        self.save_param_path = self.save_param_path + new_folder_name + '/'
        os.makedirs(self.save_param_path, exist_ok=True)

    def data_prepare(self, data_root):
        # if not os.path.exists(data_root+"frame_fea/"):
        #     vgg19_extractor(data_root, data_root+"images/", data_root+"frame_fea/")

        if not os.path.exists(data_root + "video_fea/"):
            c3d_feature_extractor(data_root, data_root + "videos/", data_root + "video_fea/")

        if not os.path.exists(data_root + "audio_fea/"):
            os.makedirs(data_root + "audio_fea/")
            model_vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
            model_vggish.eval()

            audio_list = os.listdir(data_root + "audios/")
            for audio in audio_list:
                audio_fea_name = data_root + "audio_fea/" + '{}.pkl'.format(os.path.splitext(audio)[0])
                if os.path.exists(audio_fea_name):
                    print(audio_fea_name)
                else:
                    audio_fea = model_vggish.forward(data_root + "audios/" + audio)
                    with open(audio_fea_name, 'wb') as f:
                        pickle.dump(audio_fea, f)



    def main(self):
        collate_fn = None

        self.data_prepare(self.data_root)

        print('-' * 50)
        print('start training')
        print('-' * 50)
        results = {"video_only": {'pred': [], 'label': [], 'mask': (1, 0, 0)},
                   "text_only": {'pred': [], 'label': [], 'mask': (0, 1, 0)},
                   "audio_only": {'pred': [], 'label': [], 'mask': (0, 0, 1)},
                   "video_text": {'pred': [], 'label': [], 'mask': (1, 1, 0)},
                   "video_audio": {'pred': [], 'label': [], 'mask': (1, 0, 1)},
                   "text_audio": {'pred': [], 'label': [], 'mask': (0, 1, 1)},
                   "all_modalities": {'pred': [], 'label': [], 'mask': (1, 1, 1)}}
        for repeat in range(self.random_repeat):
            dataset = self.get_data(data_root=self.data_root, data=self.data)

            model = self.get_model()

            trainer = Trainer(model=model,
                              model_cfg=self.model_cfg,
                              dataset=dataset,
                              num_workers=self.num_workers,
                              save_param_path=self.save_param_path,
                              config=self.config,
                              results=copy.deepcopy(results),
                              repeat=repeat
                              )
            one_repeat_result = trainer.train()
            for name in results.keys():
                results[name]['pred'] = results[name]['pred'] + one_repeat_result[name]['pred']
                results[name]['label'] = results[name]['label'] + one_repeat_result[name]['label']

        for name in results.keys():
            pred = results[name]['pred']
            label = results[name]['label']
            print(f"Results for {name}:")
            acc, report, cm = get_confusionmatrix_fnd(np.array(pred), np.array(label))

        # else:
        #     print ("Not Available")

    def baseline(self):
        collate_fn = None

        self.data_prepare(self.data_root)

        print('-' * 50)
        print('start training')
        print('-' * 50)
        # model = FakeSVModel(model_cfg=self.model_cfg,
        #                     dropout=self.dropout, )

        results = {"video_only": {'pred': [], 'label': [], 'mask': (1, 0, 0)},
                   "text_only": {'pred': [], 'label': [], 'mask': (0, 1, 0)},
                   "audio_only": {'pred': [], 'label': [], 'mask': (0, 0, 1)}}

        self.config['shielding_prob'] = 0

        for name, value in results.items():
            for repeat in range(self.random_repeat):
                dataset = self.get_data(data_root=self.data_root, data=self.data)

                base_video_model = bC3D(video_dim=self.model_cfg['video_dim'], fea_dim=self.model_cfg['dim'],
                                        num_heads=self.model_cfg['num_heads'])

                base_text_model = bBert(base_model=self.model_cfg['base_model'], text_dim=self.model_cfg['text_dim'],
                                        fea_dim=self.model_cfg['dim'], dropout=self.dropout)

                base_audio_model = bVggish(fea_dim=self.model_cfg['dim'])

                base_models = {
                    "video_only": base_video_model,
                    "text_only": base_text_model,
                    "audio_only": base_audio_model,
                }

                trainer = Trainer(model=base_models[name],
                                  model_cfg=self.model_cfg,
                                  dataset=dataset,
                                  num_workers=self.num_workers,
                                  save_param_path=self.save_param_path,
                                  config=self.config,
                                  results=copy.deepcopy({name: value})
                                  )

                modality_results = trainer.train()

                results[name]['pred'] = results[name]['pred'] + modality_results[name]['pred']
                results[name]['label'] = results[name]['label'] + modality_results[name]['label']

        for name in results.keys():
            pred = results[name]['pred']
            label = results[name]['label']
            print(f"Results for {name}:")
            acc, report, cm = get_confusionmatrix_fnd(np.array(pred), np.array(label))

if __name__ == '__main__':
        main()
