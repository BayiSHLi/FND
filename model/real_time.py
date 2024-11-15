import torch
import torch.nn as nn
import torch.nn.functional as F

from video_swin_transformer import SwinTransformer3D
from transformers import AutoConfig, BertModel, AutoModelForSequenceClassification


class RealTimeModel(torch.nn.Module):
    def __init__(self,
                 model_name,
                 base_model,
                 text_dim,
                 img_dim,
                 video_dim,
                 audio_dim,
                 num_frames,
                 num_audioframes,
                 num_comments,
                 dim,
                 num_heads,
                 dropout=0.3):
        super(RealTimeModel, self).__init__()

        self.base_model = base_model
        self.text_bert = AutoModelForSequenceClassification.from_pretrained(self.base_model).requires_grad_(False)

        self.text_dim = text_dim
        self.img_dim = img_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.num_frames = num_frames
        self.num_audioframes = num_audioframes
        self.num_comments = num_comments
        self.dim = dim
        self.num_heads = num_heads

        self.video_extractor = SwinTransformer3D()
        self.audio_extractor =


        self.dropout = dropout

        # self.attention = Attention(dim=self.dim,heads=4,dropout=dropout)


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
