# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import json
import torchvision.models as models
from tqdm import tqdm, trange
from pathlib import Path
import os
from reformer_pytorch import Reformer as RF
from reformer_pytorch.reformer_pytorch import FixedPositionalEmbedding
from reformer_pytorch.reformer_pytorch import AbsolutePositionalEmbedding
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import random
import sys
from transformers import AdamW,get_cosine_schedule_with_warmup
from divpresreward import compute_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device="cpu"

class time_googlenet_embeding_loss():
    def __init__(self, type=1):
        self.FC = list(models.googlenet(pretrained=True).children())[-1]
        self.FC.eval()
        for param in self.FC.parameters():
            param.requires_grad = False
        self.FC = self.FC.to(device)
        self.type = type

        self.CosSim1 = torch.nn.CosineSimilarity(dim=1)

        # self.CosSim2=torch.nn.CosineEmbeddingLoss()

        self.CosSim2 = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, x, y, mask):
        with torch.set_grad_enabled(False):

            y = torch.max(self.FC(y), dim=-1)[1].masked_fill(~mask, 0)
            if self.type == 1:
                x = torch.max(self.FC(x), dim=-1)[1].masked_fill(~mask, 0)
                return 1 - self.CosSim1(x.float(), y.float()).mean()
            elif self.type == 2:

                return self.CosSim2(x.transpose(1, 2), y).masked_select(mask).mean()


class time_embeding_loss():
    def __init__(self, type=0):
        self.Loss = torch.nn.CosineSimilarity(dim=-1)
        self.type = type

    def __call__(self, x, y, mask):
        if self.type:
            return 1 - self.Loss(x, y).masked_select(mask)
        else:
            return 1 - self.Loss(x, y).masked_select(mask).mean()


class loss_fn_ignore():
    def __init__(self, loss_fn, type=0):
        self.loss_fn = loss_fn
        self.type = type

    def __call__(self, x, y, mask):
        loss = self.loss_fn(x, y)
        if self.type:
            return loss.masked_select(mask.unsqueeze(-1).expand_as(loss))
        else:
            return loss.masked_select(mask.unsqueeze(-1).expand_as(loss)).mean()


class pos_enc(nn.Module):
    def __init__(self, hidden_size, max_seq_len, block=0):
        super(pos_enc, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        if block == 1:
            self.pos_emb = AbsolutePositionalEmbedding(hidden_size, max_seq_len)
        elif block == 2:
            self.pos_emb = FixedPositionalEmbedding(hidden_size)

        else:
            self.pos_emb = None

    def forward(self, x):
        if self.pos_emb is not None:
            x = x + self.pos_emb(x)
            x = self.norm(x)

        return x


class lossAd():
    def __init__(self, LList):
        self.LLoss = LList

    def __call__(self, *args):
        loss = 0
        for elm in self.LLoss:
            loss += elm(*args)
        return loss


class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None, max_seq_len=256, logger=None,
                 dataset_lens={'train': None, 'test': None},testset2=None):
        """Class that Builds, Trains and Evaluates AC-SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_seq_len = max_seq_len
        self.testset2=testset2
        # or 'mean'?
        # self.MSEloss = torch.nn.MSELoss(reduction='none')
        self.losses_fn = {'L1': loss_fn_ignore(torch.nn.L1Loss(reduction='none')),
                          'MSE': loss_fn_ignore(torch.nn.MSELoss(reduction='none')),
                          'CB': lossAd([loss_fn_ignore(torch.nn.L1Loss(reduction='none')), time_embeding_loss()]),
                          # 'emb_gl_cs':(time_googlenet_embeding_loss(type=1)),
                          # 'emb_gl_ce': (time_googlenet_embeding_loss(type=2)),
                          'CosS': time_embeding_loss(),
                          }
        self.losses_fn_dist = {'L1': loss_fn_ignore(torch.nn.L1Loss(reduction='none'), type=1),
                               'MSE': loss_fn_ignore(torch.nn.MSELoss(reduction='none'), type=1),
                               # 'emb_gl_cs':(time_googlenet_embeding_loss(type=1)),
                               # 'emb_gl_ce': (time_googlenet_embeding_loss(type=2)),
                               'CosS': time_embeding_loss(type=1),
                               }
        self.Loss = self.losses_fn[config.losstype]
        self.losses_num_tr = {}
        self.losses_num_te = {}
        self.st = ''
        for k, v in self.losses_fn.items():
            self.st = self.st + ',' + k + '_train'
            self.st = self.st + ',' + k + '_test'
            self.losses_num_te[k] = 0
            self.losses_num_tr[k] = 0
        self.st = self.st + ',bestloss,Lr'
        logger.info(self.st)
        self.logger = logger
        self.best_loss = 100
        self.dataset_lens = dataset_lens
        self.tb_writer = SummaryWriter(config.name2)

    def build(self):

        # Build Modules

        if self.config.input_size != self.config.hidden_size:
            self.linear_compress_dw = nn.Linear(
                self.config.input_size,
                self.config.hidden_size).to(device)
            self.linear_compress_up = nn.Linear(
                self.config.hidden_size,
                self.config.input_size).to(device)
        else:
            self.linear_compress_dw = torch.nn.Identity().to(device)
            self.linear_compress_up = torch.nn.Identity().to(device)

        self.Transformer = RF(dim=self.config.hidden_size, depth=self.config.num_layers,
                              heads=self.config.num_heads, use_full_attn=self.config.F_atn).to(device)
        self.pos_emb = pos_enc(self.config.hidden_size, self.max_seq_len, self.config.posE).to(device)

        self.model = nn.ModuleList([self.linear_compress_dw, self.pos_emb, self.Transformer, self.linear_compress_up])
        # self.model=self.model.to(device)
        # if self.config.posE == 1:
        #     self.pos_emb = AbsolutePositionalEmbedding(self.config.hidden_size, self.max_seq_len)
        # elif self.config.posE == 2:
        #     self.pos_emb = FixedPositionalEmbedding(self.config.hidden_size)

        # if self.config.mode == 'train':
        #     # Build Optimizers

        if self.config.opt==1:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.lr
            )
        if self.config.Sc==2:
            self.Lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=100,
                                                                           cooldown=100, min_lr=0.000001)
        #elif self.config.Sc==1:
        #    self.Lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,num_warmup_steps=self.config.n_epochs*4/10,num_training_steps=self.config.n_epochs*4)
        elif self.config.Sc == 1:
            self.Lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                                num_warmup_steps=self.config.n_epochs * 4 / 10,
                                                                num_training_steps=self.config.n_epochs * 4)
        else:
            self.Lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=100,
                                                                           cooldown=100, min_lr=0.000001)

    def reconstruction_loss(self, h_origin, h_sum):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""

        return torch.norm(h_origin - h_sum, p=2)

    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        return 0.5 * torch.sum(-1 + log_variance.exp() + mu.pow(2) - log_variance)

    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""

        return torch.abs(torch.mean(scores) - self.config.regularization_factor)

    def train(self):

        step = 0
        for epoch_i in range(self.config.n_epochs):

            self.model.train()

            # running_loss = 0

            for k, v in self.losses_num_tr.items():
                self.losses_num_tr[k] = 0
            for Masked_seq, Attn_mask, target_seq, lmask,_,_ in self.train_loader:

                input_seq = Masked_seq.to(device)
                attn_mask = Attn_mask.to(device)
                target_seq = target_seq.to(device)
                lmask = lmask.to(device)

                #
                # if self.config.verbose:
                #     tqdm.write('Training ...')
                self.optimizer.zero_grad()

                input_seq_dw = self.linear_compress_dw(input_seq)
                input_seq_dw_ps = self.pos_emb(input_seq_dw)
                pred_seq_dw_ps = self.Transformer(input_seq_dw_ps, input_mask=attn_mask)
                pred_seq_dw_ps_up = self.linear_compress_up(pred_seq_dw_ps)

                loss = self.Loss(pred_seq_dw_ps_up, target_seq, attn_mask * (lmask != 0))
                loss.backward()

                for k, v in self.losses_fn.items():
                    self.losses_num_tr[k] += v(pred_seq_dw_ps_up, target_seq,
                                               attn_mask * (lmask != 0)).item() * input_seq.size(0)

                # running_loss_L1 += loss.item() * input_seq.size(0)
                # running_loss_MSE += loss_mse.item() * input_seq.size(0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

            # running_loss_L1 = running_loss_L1 / self.dataset_lens['train']
            # running_loss_MSE= running_loss_MSE / self.dataset_lens['train']
            for k, v in self.losses_num_tr.items():
                self.losses_num_tr[k] = v / self.dataset_lens['train']

            self.evaluate(epoch_i)
            # self.score(epoch_i)
        out_dict = torch.load(self.config.name + '.pth')
        out_record = list([out_dict['epoch'], out_dict['tr_Loss'], out_dict['te_Loss']])
        self.tb_writer.add_hparams(
            {"lr": self.config.lr,
             "hidden_size": self.config.hidden_size,
             "CombD":self.config.CombD,
             "batch_size": self.config.batch_size,
             "num_layers": self.config.num_layers,
             "num_heads": self.config.num_heads,
             "positional_enc": self.config.posE,
             "window_s": self.config.window_s,
             "Tr_seq_Len": self.config.Tr_seq_Len,
             "mask_ratio": self.config.mask_ratio,
             "mask_chance": self.config.mask_chance, },
            {
                "L1": out_dict['te_Loss']["L1"],
                "MSE": out_dict['te_Loss']["MSE"],
                "CosS": out_dict['te_Loss']["CosS"],
                "CB": out_dict['te_Loss']["CB"]
            },
        )

        self.tb_writer.close()
        # self.score(10)
        return out_record

    def evaluate(self, epoch_i):

        self.model.eval()
        if self.config.verbose:
            tqdm.write('Evaluting ...')
        for k, v in self.losses_num_te.items():
            self.losses_num_te[k] = 0
        running_loss = 0
        loss_dis = {}
        for k, v in self.losses_fn_dist.items():
            loss_dis[k] = list([])
        for Masked_seq, Attn_mask, target_seq, lmask,_,_ in self.test_loader:

            input_seq = Masked_seq.to(device)
            attn_mask = Attn_mask.to(device)
            target_seq = target_seq.to(device)
            lmask = lmask.to(device)

            with torch.set_grad_enabled(False):
                input_seq_dw = self.linear_compress_dw(input_seq)
                input_seq_dw_ps = self.pos_emb(input_seq_dw)
                pred_seq_dw_ps = self.Transformer(input_seq_dw_ps, input_mask=attn_mask)
                pred_seq_dw_ps_up = self.linear_compress_up(pred_seq_dw_ps)
                for k, v in self.losses_fn.items():
                    self.losses_num_te[k] += v(pred_seq_dw_ps_up, target_seq,
                                               attn_mask * (lmask != 0)).item() * input_seq.size(0)
                    if ((epoch_i + 1) % 100 == 0 or epoch_i==0) and (k in self.losses_fn_dist.keys()):
                        loss_dis[k].append(
                            self.losses_fn_dist[k](pred_seq_dw_ps_up, target_seq, attn_mask * (lmask != 0)))
                loss = self.Loss(pred_seq_dw_ps_up, target_seq, attn_mask * (lmask != 0))
                # loss = self.Loss(pred_seq_dw_ps_up, target_seq, attn_mask )
                running_loss += loss.item() * input_seq.size(0)
        for k, v in self.losses_num_te.items():
            self.losses_num_te[k] = v / self.dataset_lens['test']
        running_loss = running_loss / self.dataset_lens['test']
        if self.config.Sc==2:
            self.Lr_scheduler.step(running_loss)
            self.Lr_scheduler.step(running_loss)
        elif self.config.Sc==1:
            self.Lr_scheduler.step(epoch_i)
        if running_loss <= self.best_loss:
            out_dict = {'epoch': epoch_i,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler': self.Lr_scheduler.state_dict(),
                        'Linear_dw': self.linear_compress_dw.state_dict(),
                        'Linear_up': self.linear_compress_up.state_dict(),
                        'Transformer': self.Transformer.state_dict(),
                        'tr_Loss': self.losses_num_tr,
                        'te_Loss': self.losses_num_te,
                        }
            torch.save(out_dict, self.config.name + '.pth')
            tqdm.write('best loss updated')
            self.best_loss = running_loss

        st = ''
        for k, v in self.losses_num_tr.items():
            st = st + ',{}'.format(self.losses_num_tr[k])
            st = st + ',{}'.format(self.losses_num_te[k])
        st = st + ',{},{}'.format(self.best_loss, self.optimizer.param_groups[0]['lr'])
        self.logger.info(st)
        for k, v in zip(self.st.split(',')[1:], st.split(',')[1:]):
            self.tb_writer.add_scalar(k, float(v), epoch_i)
        self.tb_writer.add_scalar('Lr',self.optimizer.param_groups[0]['lr'],epoch_i)
        if (epoch_i + 1) % 100 == 0 or epoch_i==0:
            for k, v in loss_dis.items():
                self.tb_writer.add_histogram(k, torch.cat(v), epoch_i)
        # self.logger.info(
        #     ',{},{},{},{}'.format(tr_loss, running_loss, self.best_loss, self.optimizer.param_groups[0]['lr']))
        tqdm.write(self.st)
        tqdm.write(st)
        # score_save_path = self.config.score_dir.joinpath(
        #     f'{self.config.video_type}_{epoch_i}.json')
        # if not os.path.exists(os.path.split(score_save_path)[0]):
        #     os.makedirs(os.path.split(score_save_path)[0])
        # with open(score_save_path, 'w') as f:
        #     if self.config.verbose:
        #         tqdm.write(f'Saving score at {str(score_save_path)}.')
        #     json.dump(out_dict, f)
        # score_save_path.chmod(0o777)
        # return list([out_dict['epoch'], out_dict['tr_Loss'],out_dict['tr_Loss']])
    def score(self,iterN):
        out_dict = torch.load(self.config.name + '.pth')
        self.linear_compress_dw.load_state_dict(out_dict['Linear_dw'])
        self.linear_compress_up.load_state_dict(out_dict['Linear_up'])
        self.Transformer.load_state_dict(out_dict['Transformer'])
        self.model.eval()
        if self.config.verbose:
            tqdm.write('scoring ...')
        vid_scores_L={}
        vid_scores_R= {}
        # vid_scores_RL = {}

        for tt in range(iterN):
            self.testset2.update(random.SystemRandom().randint(0,sys.maxsize))
            #self.testset2.update(tt*5684)
            test_loader=torch.utils.data.DataLoader(self.testset2,
                                       batch_size=self.config.batch_size, shuffle=False, num_workers=0)
            for Masked_seq, Attn_mask, target_seq, lmask,vid_ind,mask_ind in test_loader:
                for elm in vid_ind:
                    if elm not in vid_scores_L.keys():
                        vid_scores_L[elm]= {}
                        vid_scores_R[elm] = {}
                        # vid_scores_RL[elm]={}
                input_seq = Masked_seq.to(device)
                attn_mask = Attn_mask.to(device)
                target_seq = target_seq.to(device)
                lmask = lmask.to(device)

                with torch.set_grad_enabled(False):
                    input_seq_dw = self.linear_compress_dw(input_seq)
                    input_seq_dw_ps = self.pos_emb(input_seq_dw)
                    pred_seq_dw_ps = self.Transformer(input_seq_dw_ps, input_mask=attn_mask)
                    pred_seq_dw_ps_up = self.linear_compress_up(pred_seq_dw_ps)
                    for i in range(pred_seq_dw_ps_up.shape[0]):

                        loss= self.Loss(pred_seq_dw_ps_up[i,...].unsqueeze(0), target_seq[i,...].unsqueeze(0), attn_mask[i,...].unsqueeze(0) * (lmask[i,...] != 0).unsqueeze(0))
                        temp_mask=(1-lmask[i,...].unsqueeze(0))*attn_mask[i,...].unsqueeze(0)

                        if self.config.invert_loss==0:
                            Reward = compute_reward(target_seq[i, ...].unsqueeze(0), temp_mask)
                        else:
                            Reward = compute_reward(target_seq[i, ...].unsqueeze(0), temp_mask)
                        # Reward=compute_reward(target_seq[i,...].unsqueeze(0),lmask[i,...].unsqueeze(0))
                        for ind_m in mask_ind[i]:
                            if ind_m.item() not in vid_scores_L[vid_ind[i]].keys():
                                vid_scores_L[vid_ind[i]][ind_m.item()]=[]
                                vid_scores_R[vid_ind[i]][ind_m.item()] = []
                                # vid_scores_RL[vid_ind[i]][ind_m.item()] = []
                            # vid_scores[vid_ind[i]][ind_m.item()].append(loss.item())
                            vid_scores_L[vid_ind[i]][ind_m.item()].append(loss.item())
                            vid_scores_R[vid_ind[i]][ind_m.item()].append(Reward.item())
                            # vid_scores_RL[vid_ind[i]][ind_m.item()].append(Reward.item() + loss.item())

        for k,v in vid_scores_L.items():
            temp_L=np.zeros(max(vid_scores_L[k].keys()))
            temp_R = np.zeros(max(vid_scores_L[k].keys()))
            temp_RL = np.zeros(max(vid_scores_L[k].keys()))
            for i in range(len(temp_L)):
                temp_L[i]=(torch.mean(torch.tensor(vid_scores_L[k][i])))
                temp_R[i] = (torch.mean(torch.tensor(vid_scores_R[k][i])))
                # temp_RL[i] = (torch.mean(torch.tensor(vid_scores_RL[k][i])))

                # temp[i] = -torch.mean(torch.tensor(vid_scores[k][i]))
                # temp[i]=torch.sigmoid(40*(0.07-torch.mean(torch.tensor(vid_scores[k][i]))))
                # vid_scores[k]=list(temp)
                #temp[i] = -torch.mean(torch.tensor(vid_scores[k][i]))

            # vid_scores[k] = list((temp - temp.min()) / (temp.max() - temp.min()))
            vid_scores_L[k]=list(temp_L)
            vid_scores_R[k] = list(temp_R)
            # vid_scores_RL[k] = list(temp_RL)
                # loss = self.Loss(pred_seq_dw_ps_up, target_seq, attn_mask )
        with open(self.config.name + '_' + ','.join([str(x) for x in self.config.mask_ratio_fs]) +",dwsfs_{}".format(self.config.dw_s_fs)+",IV_{}".format(self.config.invert_loss)+'_L.json', 'w') as f:
            # if self.config.verbose:
            #     tqdm.write(f'Saving score at {str(score_save_path)}.')
            json.dump(vid_scores_L, f)
        with open(self.config.name + '_' + ','.join([str(x) for x in self.config.mask_ratio_fs]) +",dwsfs_{}".format(self.config.dw_s_fs)+",IV_{}".format(self.config.invert_loss)+'_R.json', 'w') as f:
            # if self.config.verbose:
            #     tqdm.write(f'Saving score at {str(score_save_path)}.')
            json.dump(vid_scores_R, f)
        # with open(self.config.name +'_RL.json', 'w') as f:
        #     # if self.config.verbose:
        #     #     tqdm.write(f'Saving score at {str(score_save_path)}.')
        #     json.dump(vid_scores_RL, f)
        # score_save_path.chmod(0o777)
        return out_dict['te_Loss']

if __name__ == '__main__':
    pass
