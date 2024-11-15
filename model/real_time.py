import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, BertModel, AutoModelForSequenceClassification
from collections import OrderedDict
from .video_swin_transformer import SwinTransformer3D
from .clap import PretrainedCLAP
from .decision_net import DecisionNet
from .multi_modal_fusion import co_attention

class RealTimeModel(torch.nn.Module):
    def __init__(self,
                 model_name,
                 embed_dim,
                 base_model,
                 pretrained_video_model,
                 video_net_params,
                 audio_net_param,
                 text_dim,
                 lstm_len,
                 dropout=0.3):
        super(RealTimeModel, self).__init__()

        self.model_name = model_name
        self.embed_dim = embed_dim
        self.base_model = base_model
        self.text_bert = AutoModelForSequenceClassification.from_pretrained(self.base_model).requires_grad_(False)

        self.video_extractor = SwinTransformer3D(**video_net_params)
        checkpoint = torch.load(pretrained_video_model)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v
        self.video_extractor.load_state_dict(new_state_dict)

        self.audio_extractor = PretrainedCLAP(**audio_net_param)

        self.dropout = dropout
        self.video_dim = video_net_params['embed_dim']
        self.audio_dim = audio_net_param['final_feat_dim']
        self.text_dim = text_dim

        self.co_attention_av = co_attention(d_k=self.video_dim,
                                            d_v=self.audio_dim,
                                            dropout=self.dropout,)

        self.co_attention_text = co_attention(d_k=self.text_dim,
                                            d_v=self.video_dim,
                                            dropout=self.dropout,)
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim, num_layers=lstm_len)

        self.decision_net = DecisionNet(input_dim=self.video_dim, hidden_dim=self.embed_dim, output_dim=self.embed_dim)

        self.dropout = dropout

        self.video_bn = nn.BatchNorm1d(self.embed_dim)
        self.text_bn = nn.BatchNorm1d(self.embed_dim)
        self.audio_bn = nn.BatchNorm1d(self.embed_dim)

        self.classifier = nn.Linear(self.embed_dim, 2)

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
