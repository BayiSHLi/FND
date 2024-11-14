import copy
import os
import time
from tqdm import tqdm

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchinfo import summary
import torch.nn as nn
from utils.metrics import *
import wandb


class Trainer():
    def __init__(self,
                 model,
                 model_cfg,
                 train_set,
                 test_set,
                 mode,
                 exp_name,
                 batch_size,
                 num_workers,
                 save_param_path,
                 device,
                 epochs,
                 epoch_stop,
                 lr,
                 weight_decay,
                 shielding_prob,
                 average_weight,
                 save_threshold = 0.0,
                 start_epoch = 0,
                 results = None,
                 repeat = 0,
                 ):
        self.model_cfg = model_cfg
        self.model = model
        self.train_set = train_set
        self.test_set = test_set

        self.mode = mode

        self.num_workers = num_workers
        if not os.path.exists(save_param_path):
            os.makedirs(save_param_path)
        self.save_param_path = save_param_path
        self.start_epoch = start_epoch
        self.save_threshold = save_threshold
        self.repeat = repeat
        self.exp_name = exp_name

        self.batch_size = batch_size

        self.device = device
        self.num_epochs = epochs
        self.epoch_stop = epoch_stop
        self.lr = lr
        self.weight_decay = weight_decay
        self.p = shielding_prob
        self.average_weight = average_weight

        if self.model:
            self.model_init()

        self.criterion = nn.CrossEntropyLoss()

        self.num_folds = 5
        self.kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        self.results = results

    def FIDES_collate_fn(self, batch):
        num_comments = self.model_cfg['num_comments']
        num_frames = self.model_cfg['num_frames']
        num_audioframes = self.model_cfg['num_audioframes']

        speech_inputid = [item['speech_inputid'] for item in batch]
        speech_mask = [item['speech_mask'] for item in batch]

        caption_inputid = [item['caption_inputid'] for item in batch]
        caption_mask = [item['caption_mask'] for item in batch]

        comments_inputid = [item['comments_inputid'] for item in batch]
        comments_mask = [item['comments_mask'] for item in batch]

        c3dfea = [item['c3dfea'] for item in batch]
        c3dfea, _ = pad_frame_sequence(num_frames, c3dfea)

        # framesfea = [item['framesfea'] for item in batch]
        # framesfea, framesfea_masks = pad_frame_sequence(num_frames, framesfea)
        #
        audiofea = [item['audiofea'] for item in batch]
        audiofea, _ = pad_frame_sequence(num_audioframes, audiofea)

        label = [item['label'] for item in batch]

        return {
            'label': torch.stack(label),
            'speech_inputid': torch.stack(speech_inputid),
            'speech_mask': torch.stack(speech_mask),
            'caption_inputid': torch.stack(caption_inputid),
            'caption_mask': torch.stack(caption_mask),
            'comments_inputid': torch.stack(comments_inputid),
            'comments_mask': torch.stack(comments_mask),
            'audiofea': audiofea,
            # 'audiofea_masks': audiofea_masks,
            # 'framesfea': framesfea,
            # 'framesfea_masks': framesfea_masks,
            'c3dfea': c3dfea,
            # 'c3dfea_masks': c3dfea_masks,
        }

    def random_mask(self, batch):
        batch_data = batch

        label = batch_data['label']

        mask_video = (torch.rand(batch_data['c3dfea'].shape[0]) > self.p).float()
        mask_audio = (torch.rand(batch_data['audiofea'].shape[0]) > self.p).float()
        mask_text = (torch.rand(batch_data['speech_inputid'].shape[0]) > self.p).int()

        batch_data['c3dfea'] = batch_data['c3dfea'] * mask_video.unsqueeze(1).unsqueeze(2)
        batch_data['audiofea'] = batch_data['audiofea'] * mask_audio.unsqueeze(1).unsqueeze(2)
        batch_data['speech_inputid'] = batch_data['speech_inputid'] * mask_text.unsqueeze(1)
        batch_data['caption_inputid'] = batch_data['caption_inputid'] * mask_text.unsqueeze(1)
        batch_data['comments_inputid'] = batch_data['comments_inputid'] * mask_text.unsqueeze(1)

        return batch_data

    def model_init(self):
        optimizer_params = {
            'lr': self.lr, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': self.weight_decay,
        }
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_params)

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                 milestones=[30, 50], gamma=0.1)


    def train(self):
        wandb.init(
            # set the wandb project where this run will be logged
            project = 'FakeSV',
            name=self.save_param_path,

            # track hyperparameters and run metadata
            config={
                "learning_rate": self.lr,
                "epochs": self.num_epochs,
                "shielding_prob": self.p,
            }
        )

        since = time.time()

        self.model.cuda()


        fold_accuracies = []
        if self.average_weight:
            best_model_weights  = []

        print('-' * 10)
        print('TRAIN')
        print('-' * 10)

        for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.dataset['train'])):
            print(f'Fold {fold + 1}/{self.num_folds}')

            train_subset = Subset(self.dataset['train'], train_idx)
            val_subset = Subset(self.dataset['train'], val_idx)

            train_dataloader = DataLoader(train_subset, batch_size=self.batch_size,
                                          num_workers=self.num_workers,
                                          pin_memory=True,
                                          shuffle=True,
                                          collate_fn=self.FIDES_collate_fn)

            val_dataloader = DataLoader(val_subset, batch_size=self.batch_size,
                                         num_workers=self.num_workers,
                                         pin_memory=True,
                                         shuffle=False,
                                         collate_fn=self.FIDES_collate_fn)

            best_model_wts_test = copy.deepcopy(self.model.state_dict())
            best_acc_val = 0.0
            best_epoch_val = 0
            is_earlystop = False

            for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
                if is_earlystop:
                    break
                print('-' * 50)
                print('Epoch {}/{}'.format(epoch+1, self.start_epoch+self.num_epochs))
                print('-' * 50)

                self.model.train()
                running_loss_fnd = 0.0
                running_loss = 0.0
                tpred = []
                tlabel = []

                for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                    batch_data = self.random_mask(batch)
                    for k,v in batch_data.items():
                        batch_data[k]=v.cuda()
                    label = batch_data['label']
                    speech_inputid = batch_data['speech_inputid']
                    speech_mask = batch_data['speech_mask']
                    caption_inputid = batch_data['caption_inputid']  # (batch,512)
                    caption_mask = batch_data['caption_mask']  # (batch,512)
                    comments_inputid = batch_data['comments_inputid']  # (batch,250)
                    comments_mask = batch_data['comments_mask']  # (batch,250)
                    c3dfea = batch_data['c3dfea']  # (batch, 36, 4096)
                    audiofea = batch_data['audiofea']  # (batch,36,12288)

                    if fold==0 and epoch==0 and batch_idx==0:
                        summary(self.model, input_data=(speech_inputid, speech_mask,
                                                        caption_inputid, caption_mask,
                                                        comments_inputid, comments_mask,
                                                        c3dfea, audiofea))

                    outputs,fea = self.model(speech_inputid, speech_mask, caption_inputid,
                                                        caption_mask, comments_inputid, comments_mask, c3dfea, audiofea)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, label)

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    wandb.log({"loss": loss})
                    # wandb.log({"loss": loss, "learning rate": self.optimizer.param_groups[0]['lr']})

                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * label.size(0)

                epoch_loss = running_loss / len(train_dataloader.dataset)

                wandb.log({"learning rate": self.optimizer.param_groups[0]['lr']})
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                print('Loss: {:.4f} '.format(epoch_loss))

                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                vpred = []
                vlabel = []
                with torch.no_grad():
                    for batch in tqdm(val_dataloader):
                        batch_data = self.random_mask(batch)
                        for k, v in batch_data.items():
                            batch_data[k] = v.cuda()
                        label = batch_data['label']
                        speech_inputid = batch_data['speech_inputid']
                        speech_mask = batch_data['speech_mask']
                        caption_inputid = batch_data['caption_inputid']  # (batch,512)
                        caption_mask = batch_data['caption_mask']  # (batch,512)
                        comments_inputid = batch_data['comments_inputid']  # (batch,250)
                        comments_mask = batch_data['comments_mask']  # (batch,250)
                        c3dfea = batch_data['c3dfea']  # (batch, 36, 4096)
                        audiofea = batch_data['audiofea']  # (batch,36,12288)

                        outputs, fea = self.model(speech_inputid, speech_mask, caption_inputid,
                                                        caption_mask, comments_inputid, comments_mask, c3dfea, audiofea)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, label)

                        val_loss += loss.item() * label.size(0)

                        vlabel.extend(label.detach().cpu().numpy().tolist())
                        vpred.extend(preds.detach().cpu().numpy().tolist())

                    results = metrics(vlabel, vpred)


                print(results)

                wandb.log({"acc": results['acc'],
                           "f1": results['f1'],
                           "precision": results['precision'],
                           "recall": results['recall'],
                           "auc": results['auc'],
                           })


                if results['acc'] > best_acc_val:
                    best_acc_val = results['acc']
                    best_model_wts_test = copy.deepcopy(self.model.state_dict())
                    best_epoch_val = epoch+1
                    if best_acc_val > self.save_threshold:
                        save_path = self.save_param_path + "repeat_" + str(self.repeat) + '/'
                        os.makedirs(save_path, exist_ok=True)
                        save_name = save_path + "epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val)
                        torch.save(self.model.state_dict(), save_name)
                        print ("saved " + save_name)
                else:
                    if epoch-best_epoch_val >= self.epoch_stop-1:
                        is_earlystop = True
                        print ("early stopping...")

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print("Best model: " + save_name)

            fold_accuracies.append(best_acc_val)
            if self.average_weight:
                best_model_weights.append(best_model_wts_test)
            print(f'Fold {fold + 1} Accuracy: {best_acc_val:.4f}')

        avg_accuracy = np.mean(fold_accuracies)
        print(f'Average Accuracy: {avg_accuracy:.4f}')

        if self.average_weight:
            averaged_weights = {}
            for key in best_model_weights[0].keys():
                averaged_weights[key] = torch.mean(
                    torch.stack([model_weights[key].float() for model_weights in best_model_weights]), dim=0)
            self.model.load_state_dict(averaged_weights)
        else:
            self.model.load_state_dict(best_model_wts_test)

        return self.test()


    def test(self):
        test_dataloader = DataLoader(self.dataset['test'], batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=True,
                                     shuffle=False,
                                     collate_fn=self.FIDES_collate_fn)
        print('-' * 10)
        print('TEST')
        print('-' * 10)

        since = time.time()

        self.model.cuda()
        self.model.eval()

        print(self.exp_name)
        for name in self.results.keys():
            # print(self.results[name])
            pred = []
            label = []
            mask_v_val, mask_t_val, mask_a_val = self.results[name]['mask']
            for batch in test_dataloader:
                with torch.no_grad():
                    # TODO: to cuda
                    # TODO: can not reach test function in dubug
                    batch_data = batch

                    mask_video = torch.full((batch_data['c3dfea'].shape[0], 1, 1), mask_v_val)
                    mask_text = torch.full((batch_data['speech_inputid'].shape[0], 1), mask_t_val)
                    mask_audio = torch.full((batch_data['audiofea'].shape[0], 1, 1), mask_a_val)

                    batch_data['c3dfea'] = batch_data['c3dfea'] * mask_video
                    batch_data['speech_inputid'] = batch_data['speech_inputid'] * mask_text
                    batch_data['caption_inputid'] = batch_data['caption_inputid'] * mask_text
                    batch_data['comments_inputid'] = batch_data['comments_inputid'] * mask_text
                    batch_data['audiofea'] = batch_data['audiofea'] * mask_audio

                    for k,v in batch_data.items():
                        batch_data[k]=v.cuda()
                    batch_label = batch_data['label']

                    speech_inputid = batch_data['speech_inputid']
                    speech_mask = batch_data['speech_mask']
                    caption_inputid = batch_data['caption_inputid']  # (batch,512)
                    caption_mask = batch_data['caption_mask']  # (batch,512)
                    comments_inputid = batch_data['comments_inputid']  # (batch,250)
                    comments_mask = batch_data['comments_mask']  # (batch,250)
                    c3dfea = batch_data['c3dfea']  # (batch, 36, 4096)
                    audiofea = batch_data['audiofea']  # (batch,36,12288)

                    batch_outputs,fea = self.model(speech_inputid, speech_mask, caption_inputid,
                                                        caption_mask, comments_inputid, comments_mask, c3dfea, audiofea)

                    _, batch_preds = torch.max(batch_outputs, 1)

                    label.extend(batch_label.detach().cpu().numpy().tolist())
                    pred.extend(batch_preds.detach().cpu().numpy().tolist())

            print(f"Results for {name}:")
            print (metrics(label, pred))
            self.results[name]['pred'] = pred
            self.results[name]['label'] = label
            # print(f"Results for {name}: ")
            # acc, report, cm = get_confusionmatrix_fnd(np.array(pred), np.array(label))
            #     'name': name,
            #     'acc': acc,
            #     'report': report,
            #     'cm': cm,
            # })

        return self.results


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
