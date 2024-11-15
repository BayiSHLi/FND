import argparse
import yaml
import os
import copy
from datetime import datetime
from torch.utils.data import random_split
from model import *
from utils import *
import pickle
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='baseline-0930')
    parser.add_argument('--config', default='./config/FakeTT_RealTime.yml')
    parser.add_argument("--work_dir", default='./experiments/', type=str, help="work_dir")
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--random_repeat', default=3)
    args = parser.parse_args()

    print(args)
    return args


def init_exp(work_dir, config):
    data_cfg = config['dataset']
    model_cfg = config['model']
    exp_cfg = config['exp']
    current_date = datetime.now().strftime("%Y-%m-%d")
    save_param_path = work_dir + data_cfg["data_name"] + "_" \
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
    exp_cfg["save_param_path"] = save_param_path

    return exp_cfg, data_cfg, model_cfg


def get_data(data_cfg):
    if data_cfg['data_name'] == 'FakeTT':
        dataset = FakeTTDataset(**data_cfg)

    elif data_cfg['data_name'] == 'FIDES':
        dataset = FIDESDataset(**data_cfg)
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    return train_set, test_set



def run(random_repeat, exp_cfg, data_cfg, model_cfg):
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
    for repeat in range(random_repeat):
        train_set, test_set = get_data(data_cfg)

        if model_cfg['model_name'] == 'FakeSV':
            model = FakeSVModel(**model_cfg)

        elif model_cfg['model_name'] == 'RealTime':
            model = RealTimeModel(**model_cfg)

        trainer = Trainer(model=model,
                          model_cfg=model_cfg,
                          train_set=train_set,
                          test_set=test_set,
                          results=copy.deepcopy(results),
                          repeat=repeat,
                          **exp_cfg
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
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    exp_cfg, data_cfg, model_cfg = init_exp(args.work_dir, config)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(exp_cfg['device'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    run(args.random_repeat, exp_cfg, data_cfg, model_cfg)


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



if __name__ == '__main__':
        main()
