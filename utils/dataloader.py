import os
import pickle

import h5py
import jieba
import jieba.analyse as analyse
import numpy as np
import pandas as pd
import torch
from torchaudio import transforms as T
import torchaudio
import ffmpeg
import io
from moviepy.editor import VideoFileClip

from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from transformers import ClapProcessor
import json

def str2num(str_x):
    if isinstance(str_x, float):
        return str_x
    elif str_x.isdigit():
        return int(str_x)
    elif 'w' in str_x:
        return float(str_x[:-1])*10000
    elif 'billion' in str_x:
        return float(str_x[:-1])*100000000
    else:
        print ("error")
        print (str_x)

class FIDESDataset(Dataset):

    def __init__(self, data_name, data_root, data_file):
        self.data_root = data_root
        self.data_name = data_name

        try:
            self.data = pd.read_csv(data_root + data_file, encoding='unicode_escape')
        except:
            print(f'file {data_root + data_file} not found.')


        self.videopath = self.data_root + '/videos/'
        self.framepath = self.data_root + '/images/'
        self.audiopath = self.data_root + '/audios/'

        self.framefeapath = self.data_root + '/frame_fea/'
        self.videofeapath = self.data_root + '/video_fea/'
        self.audiofeapath = self.data_root + '/audio_fea/'

        self.vid = []

        self.tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item[0]

        # label
        label = torch.tensor(1 if item['LABEL']==1 else 0)

        # text
        speech_tokens = self.tokenizer(str(item['SPEECH']), max_length=512, padding='max_length',
                                          truncation=True)
        speech_inputid = torch.LongTensor(speech_tokens['input_ids'])
        speech_mask = torch.LongTensor(speech_tokens['attention_mask'])

        caption_tokens = self.tokenizer(str(item['CAPTION']) + " " + str(item['OST']), max_length=50,
                                        padding='max_length', truncation=True)
        caption_inputid = torch.LongTensor(caption_tokens['input_ids'])
        caption_mask = torch.LongTensor(caption_tokens['attention_mask'])

        comment_tokens = self.tokenizer(str(item['HASHTAGS']) + " " + str(item['USER']) + " " + str(item['LINK']),
                                        max_length=50, padding='max_length', truncation=True)
        comments_inputid = torch.LongTensor(comment_tokens['input_ids'])
        comments_mask = torch.LongTensor(comment_tokens['attention_mask'])

        # audio
        with open(self.audiofeapath + str(vid) + '.pkl', "rb") as fr:
            audiofea = pickle.load(fr).to('cpu')
            # audiofea_min = audiofea.min(dim=0, keepdim=True)[0]
            # audiofea_max = audiofea.max(dim=0, keepdim=True)[0]
            # audiofea = (audiofea - audiofea_min) / (audiofea_max - audiofea_min)


        # frames
        # frames = pickle.load(open(os.path.join(self.framefeapath, vid + '.pkl'), 'rb'))
        # framesfea = torch.FloatTensor(frames)

        # video
        with open(self.videofeapath + str(vid) + '.pkl', "rb") as fr:
            c3dfea = pickle.load(fr)

        return {
            'label': label,
            'speech_inputid': speech_inputid,
            'speech_mask': speech_mask,
            'audiofea': audiofea,
            # 'framesfea':None,
            'c3dfea': c3dfea,
            'caption_inputid': caption_inputid,
            'caption_mask': caption_mask,
            'comments_inputid': comments_inputid,
            'comments_mask': comments_mask,
        }


class FakeTTDataset(Dataset):
    def __init__(self, data_name, data_root, data_file, sample_rate, slice_len):
        self.data_name = data_name
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.slice_len = slice_len

        self.data_all = pd.read_json(data_root + data_file, orient='records', lines=True,
                                     dtype={'video_id': str})

        self.tokenizer = ClapProcessor.from_pretrained("laion/clap-htsat-unfused", sampling_rate=16000)

    def __len__(self):
        return self.data_all.shape[0]

    def __getitem__(self, idx):
        item = self.data_all.iloc[idx]
        vid = item['video_id']
        if item['annotation'] == 'fake':
            label = 1
        elif item['annotation'] == 'real':
            label = 0
        else:
            label = 2

        meta_info = (
            f"On {item['publish_time']}, a video was published with the description '{item['description']}'. "
            f"The user, who is {item['user_certify']}, included the following note: '{item['user_description']}'. "
            f"The event associated with this video is: {item['event']}."
        )

        inputs = self.tokenizer.tokenizer(text=meta_info, return_tensors="pt",
                                          padding="max_length", max_length=64)
        text_inputs = inputs['input_ids']
        attn_masks = inputs['attention_mask']

        video_file = os.path.join(self.data_root, 'videos', vid + '.mp4')
        keyframes, video_len = self.extract_keyframes()

        wav_file = os.path.join(self.data_root, 'audios', vid + '.wav')
        wav_data, sr = torchaudio.load(wav_file)
        if sr != self.sample_rate:
            resample = T.Resample(sr, self.sample_rate)
            wav_data = resample(wav_data)

        inputs = self.tokenizer(audios=audio_image, return_tensors="pt", sampling_rate=self.sample_rate)
        audio_inputs = inputs["input_features"]
        audio_chunk = self.slice_audio(start, end, wav_data)

    def extract_keyframes(self, video_path):

        process = (
            ffmpeg
            .input(video_path)
            .output('pipe:', vf='select=eq(pict_type\\,I)', vsync='vfr', format='image2pipe', vcodec='mjpeg')
            .run_async(pipe_stdout=True)
        )

        keyframes = []
        while True:
            # 从管道中读取图像数据
            in_bytes = process.stdout.read(1024 * 1024)  # 读取数据块
            if not in_bytes:
                break
            # 将字节流转换为图像对象
            image = Image.open(io.BytesIO(in_bytes))
            keyframes.append(image)

        process.wait()  # 等待进程结束
        return keyframes

    def slice_audio(self, start, end, wav_data):
        """
        SCL paper..
        sample_rate denotes how many sample points for one second
        """
        max_ind = wav_data.shape[1]
        start_ind = min(int(start * self.sample_rate), max_ind)
        end_ind = min(int(end * self.sample_rate), max_ind)

        return wav_data[:, start_ind: end_ind]

def split_word(df):
    title = df['description'].values
    comments = df['comments'].apply(lambda x:' '.join(x)).values
    text = np.concatenate([title, comments],axis=0)
    analyse.set_stop_words('./data/stopwords.txt')
    all_word = [analyse.extract_tags(txt) for txt in text.tolist()]
    corpus = [' '.join(word) for word in all_word]
    return corpus


def pad_frame_by_seg(seq_len, lst, seg):
    result = []
    seg_indicators = []
    sampled_seg = []
    for i in range(len(lst)):
        video = lst[i]
        v_sampled_seg = []
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        seg_video = seg[i]
        seg_len = len(seg_video)
        if seg_len >= seq_len:
            gap = seg_len // seq_len
            seg_video = seg_video[::gap][:seq_len]
            sample_index = []
            sample_seg_indicator = []
            for j in range(len(seg_video)):
                v_sampled_seg.append(seg_video[j])
                if seg_video[j][0] == seg_video[j][1]:
                    sample_index.append(seg_video[j][0])
                else:
                    sample_index.append(np.random.randint(seg_video[j][0], seg_video[j][1]))
                sample_seg_indicator.append(j)
            video = video[sample_index]
            mask = sample_seg_indicator
        else:
            if ori_len < seq_len:
                video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)

                mask = []
                for j in range(len(seg_video)):
                    v_sampled_seg.append(seg_video[j])
                    mask.extend([j] * (seg_video[j][1] - seg_video[j][0] + 1))
                mask.extend([-1] * (seq_len - len(mask)))

            else:

                sample_index = []
                sample_seg_indicator = []
                seg_len = [(x[1] - x[0]) + 1 for x in seg_video]
                sample_ratio = [seg_len[i] / sum(seg_len) for i in range(len(seg_len))]
                sample_len = [seq_len * sample_ratio[i] for i in range(len(seg_len))]
                sample_per_seg = [int(x) + 1 if x < 1 else int(x) for x in sample_len]

                sample_per_seg = [x if x <= seg_len[i] else seg_len[i] for i, x in enumerate(sample_per_seg)]
                additional_sample = sum(sample_per_seg) - seq_len
                if additional_sample > 0:
                    idx = 0
                    while additional_sample > 0:
                        if idx == len(sample_per_seg):
                            idx = 0
                        if sample_per_seg[idx] > 1:
                            sample_per_seg[idx] = sample_per_seg[idx] - 1
                            additional_sample = additional_sample - 1
                        idx += 1

                elif additional_sample < 0:
                    idx = 0
                    while additional_sample < 0:
                        if idx == len(sample_per_seg):
                            idx = 0
                        if seg_len[idx] - sample_per_seg[idx] >= 1:
                            sample_per_seg[idx] = sample_per_seg[idx] + 1
                            additional_sample = additional_sample + 1
                        idx += 1

                for seg_idx in range(len(sample_per_seg)):
                    sample_seg_indicator.extend([seg_idx] * sample_per_seg[seg_idx])

                for j in range(len(seg_video)):
                    v_sampled_seg.append(seg_video[j])
                    if sample_per_seg[j] == seg_len[j]:
                        sample_index.extend(np.arange(seg_video[j][0], seg_video[j][1] + 1))

                    else:
                        sample_index.extend(
                            np.sort(np.random.randint(seg_video[j][0], seg_video[j][1] + 1, sample_per_seg[j])))

                sample_index = np.array(sample_index)
                sample_index = np.sort(sample_index)
                video = video[sample_index]
                batch_sample_seg_indicator = np.array(sample_seg_indicator)
                mask = batch_sample_seg_indicator
                v_sampled_seg.sort(key=lambda x: x[0])

        result.append(video)
        mask = torch.IntTensor(mask)
        sampled_seg.append(v_sampled_seg)
        seg_indicators.append(mask)
    return torch.stack(result), torch.stack(seg_indicators), sampled_seg


def pad_segment(seg_lst, target_len):
    for sl_idx in range(len(seg_lst)):
        for s_idx in range(len(seg_lst[sl_idx])):
            seg_lst[sl_idx][s_idx] = torch.tensor(seg_lst[sl_idx][s_idx])
        if len(seg_lst[sl_idx]) < target_len:
            seg_lst[sl_idx].extend([torch.tensor([-1, -1])] * (target_len - len(seg_lst[sl_idx])))
        else:
            seg_lst[sl_idx] = seg_lst[sl_idx][:target_len]
        seg_lst[sl_idx] = torch.stack(seg_lst[sl_idx])

    return torch.stack(seg_lst)


def pad_unnatural_phrase(phrase_lst, target_len):
    for pl_idx in range(len(phrase_lst)):
        if len(phrase_lst[pl_idx]) < target_len:
            phrase_lst[pl_idx] = torch.cat((phrase_lst[pl_idx], torch.zeros(
                [target_len - len(phrase_lst[pl_idx]), phrase_lst[pl_idx].shape[1]], dtype=torch.long)), dim=0)
        else:
            phrase_lst[pl_idx] = phrase_lst[pl_idx][:target_len]
    return torch.stack(phrase_lst)

def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)

def collate_fn_FakeingRecipe(batch):
    num_visual_frames=83
    num_segs=83
    num_phrase=80

    vid = [item['vid'] for item in batch]
    label = torch.stack([item['label'] for item in batch])
    all_phrase_semantic_fea = [item['all_phrase_semantic_fea'] for item in batch]
    all_phrase_emo_fea = torch.stack([item['all_phrase_emo_fea'] for item in batch])

    raw_visual_frames = [item['raw_visual_frames'] for item in batch]
    raw_audio_emo = [item['raw_audio_emo'] for item in batch]
    fps = torch.stack([item['fps'] for item in batch])
    total_frame = torch.stack([item['total_frame'] for item in batch])

    content_visual_frames, _ = pad_frame_sequence(num_visual_frames,raw_visual_frames)
    raw_audio_emo = torch.cat(raw_audio_emo,dim=0)

    all_phrase_semantic_fea=[x if x.shape[0]==512 else torch.cat((x,torch.zeros([512-x.shape[0],x.shape[1]],dtype=torch.float)),dim=0) for x in all_phrase_semantic_fea] #batch*512*768
    all_phrase_semantic_fea=torch.stack(all_phrase_semantic_fea)

    ocr_pattern_fea = torch.stack([item['ocr_pattern_fea'] for item in batch])
    ocr_phrase_fea = [item['ocr_phrase_fea'] for item in batch]
    ocr_time_region = [item['ocr_time_region'] for item in batch]

    visual_time_region = [item['visual_time_region'] for item in batch]

    visual_frames_fea,visual_frames_seg_indicator,sampled_seg=pad_frame_by_seg(num_visual_frames,raw_visual_frames,visual_time_region)
    visual_seg_paded=pad_segment(sampled_seg,num_segs)

    ocr_phrase_fea=pad_unnatural_phrase(ocr_phrase_fea,num_phrase)
    ocr_time_region=pad_unnatural_phrase(ocr_time_region,num_phrase)

    return {
        'vid': vid,
        'label': label,
        'fps': fps,
        'total_frame': total_frame,
        'all_phrase_semantic_fea': all_phrase_semantic_fea,
        'all_phrase_emo_fea': all_phrase_emo_fea,
        'raw_visual_frames': content_visual_frames,
        'raw_audio_emo': raw_audio_emo,
        'ocr_pattern_fea': ocr_pattern_fea,
        'ocr_phrase_fea': ocr_phrase_fea,
        'ocr_time_region': ocr_time_region,
        'visual_frames_fea': visual_frames_fea,
        'visual_frames_seg_indicator': visual_frames_seg_indicator,
        'visual_seg_paded': visual_seg_paded
    }
