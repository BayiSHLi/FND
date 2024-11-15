import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from transformers import AutoConfig, BertModel, AutoModelForSequenceClassification

from utils.metrics import *


class FakeSVModel(torch.nn.Module):
    def __init__(self, model_cfg, dropout):
        super(FakeSVModel, self).__init__()

        self.base_model = model_cfg['base_model']

        self.text_bert = AutoModelForSequenceClassification.from_pretrained(self.base_model).requires_grad_(False)

        self.text_dim = model_cfg['text_dim']
        self.img_dim = model_cfg['img_dim']
        self.video_dim = model_cfg['video_dim']
        self.audio_dim = model_cfg['audio_dim']
        self.num_frames = model_cfg['num_frames']
        self.num_audioframes = model_cfg['num_audioframes']
        self.num_comments = model_cfg['num_comments']
        self.dim = model_cfg['dim']
        self.num_heads = model_cfg['num_heads']

        self.dropout = dropout

        # self.attention = Attention(dim=self.dim,heads=4,dropout=dropout)

        # self.co_attention_ta = co_attention(d_k=self.dim, d_v=self.dim, n_heads=self.num_heads, dropout=self.dropout,
        #                                     d_model=self.dim,
        #                                     visual_len=self.num_audioframes, sen_len=512, fea_v=self.dim,
        #                                     fea_s=self.text_dim, pos=True)
        # self.co_attention_tv = co_attention(d_k=self.dim, d_v=self.dim, n_heads=self.num_heads, dropout=self.dropout,
        #                                     d_model=self.dim,
        #                                     visual_len=self.num_frames, sen_len=512, fea_v=self.dim,
        #                                     fea_s=self.text_dim, pos=True)
        self.trm = nn.TransformerEncoderLayer(d_model=self.dim, nhead=2, batch_first=True)

        self.linear_speech = nn.Sequential(torch.nn.Linear(self.text_dim, self.dim), torch.nn.ReLU(),
                                           nn.Dropout(p=self.dropout))
        self.linear_caption = nn.Sequential(torch.nn.Linear(self.text_dim, self.dim), torch.nn.ReLU(),
                                            nn.Dropout(p=self.dropout))
        self.linear_comment = nn.Sequential(torch.nn.Linear(self.text_dim, self.dim), torch.nn.ReLU(),
                                            nn.Dropout(p=self.dropout))
        # self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, self.dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        # self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, self.dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_video = nn.Sequential(torch.nn.Linear(self.video_dim, self.dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(self.audio_dim, self.dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))

        self.video_bn = nn.BatchNorm1d(self.dim)
        self.text_bn = nn.BatchNorm1d(self.dim)
        self.audio_bn = nn.BatchNorm1d(self.dim)

        self.classifier = nn.Linear(self.dim, 2)

    def forward(self, speech_inputid, speech_mask, caption_inputid,
                caption_mask, comments_inputid, comments_mask, c3dfea, audiofea):
        ### Speech ###
        out_speech = self.text_bert.bert(speech_inputid, attention_mask=speech_mask)
        fea_speech = self.linear_speech(out_speech[1])
        fea_speech = self.text_bn(fea_speech)

        ### Caption ###
        out_caption = self.text_bert.bert(caption_inputid, attention_mask=caption_mask)  # (batch,sequence,768)
        fea_caption = self.linear_caption(out_caption[1])
        fea_caption = self.text_bn(fea_caption)

        ### Comment ###
        out_comments = self.text_bert.bert(comments_inputid, attention_mask=comments_mask)
        fea_comments = self.linear_comment(out_comments[1])  # (batch,self.dim)
        fea_comments = self.text_bn(fea_comments)

        ### C3D ###
        fea_video = self.linear_video(c3dfea)  # (batch, frames, 128)
        fea_video = self.co_attention_tv(v=fea_video, s=out_speech['last_hidden_state'], v_len=c3dfea.shape[1],
                                         s_len=out_speech['last_hidden_state'].shape[1])
        fea_video = torch.mean(fea_video, -2)
        fea_video = self.video_bn(fea_video)

        ### Audio Frames ###
        fea_audio = self.linear_audio(audiofea)
        fea_audio = self.co_attention_ta(v=fea_audio, s=out_speech['last_hidden_state'], v_len=audiofea.shape[1],
                                         s_len=out_speech['last_hidden_state'].shape[1])
        fea_audio = torch.mean(fea_audio, -2)
        fea_audio = self.audio_bn(fea_audio)

        # ### Image Frames ###
        # frames=batch_data['frames']#(batch,30,4096)
        # frames_masks = batch_data['frames_masks']
        # fea_img = self.linear_img(frames)
        # fea_speech = torch.mean(fea_speech, -2)

        fea_speech = fea_speech.unsqueeze(1)
        fea_comments = fea_comments.unsqueeze(1)
        fea_caption = fea_caption.unsqueeze(1)

        # fea_img = fea_img.unsqueeze(1)
        fea_audio = fea_audio.unsqueeze(1)
        fea_video = fea_video.unsqueeze(1)

        fea = torch.cat((fea_speech, fea_audio, fea_video, fea_caption, fea_comments), 1)  # (bs, 6, 128)

        fea = self.trm(fea)
        fea = torch.mean(fea, -2)

        output = self.classifier(fea)  # 0 fake, 1 real, 2

        return output, fea


