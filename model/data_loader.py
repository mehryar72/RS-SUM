# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
# from fragments import calculate_fragments
import itertools
import random
import sys
import ruptures as rpt
Records = {'Max_Len': 0}


class VideoData(Dataset):
    def __init__(self, mode, split_index, name='tvsum', dw_s=1, full_Len=0, Tr_seq_Len=256, Masking=1, Full_shot_mask=1,
                 mask_ratio=0.15, mask_chance=0.8, replace_chance=0.1, overlap=1, window_s=8,CombD=0,DlTyp=0,invert_loss=0,nwB=0):
        self.mode = mode
        self.name = name
        self.datasets = ['../data/SumMe/eccv16_dataset_summe_google_pool5.h5',
                         '../data/TVSum/eccv16_dataset_tvsum_google_pool5.h5',
                         '../data/ovp/eccv16_dataset_ovp_google_pool5.h5',
                         '../data/youtube/eccv16_dataset_youtube_google_pool5.h5']
        self.splits_filename = ['../data/splits/' + self.name + '_splits.json']
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)

        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]
        elif 'ovp' in self.splits_filename[0]:
            self.filename = self.datasets[2]
        elif 'youtube' in self.splits_filename[0]:
            self.filename = self.datasets[3]
        hdf = h5py.File(self.filename, 'r')
        self.action_fragments = {}
        self.list_features = []
        self.list_vid_ind=[]
        self.list_features_sb_track = []
        self.list_features_ind_track=[]
        self.Masking = Masking
        self.Full_shot_mask = Full_shot_mask
        self.mask_ratio = mask_ratio
        self.mask_chance = mask_chance
        self.replace_chance = replace_chance
        self.window_s = window_s  # use small windowsize for dw_s==1
        self.dw_s = dw_s
        self.rng=random.SystemRandom()
        self.DlTyp=DlTyp
        self.invert_loss=invert_loss
        self.Tr_seq_Len=Tr_seq_Len
        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
        # Max_len=0
        for vid_ind,video_name in enumerate(self.split[('test' if CombD else self.mode) + '_keys']):
            len1=len(self.list_features)
            features = torch.Tensor(np.array(hdf[video_name + '/features']))
            if not nwB:
                sb = np.array(hdf[video_name + '/change_points'])
                len_fe_frames = round((sb[-1, -1] + 1) / len(features))
                sb_track = np.zeros(len(features))
                for c, sb_index in enumerate(sb):
                    sb_track[range(int(np.round(sb_index[0] / len_fe_frames)),
                                   int(np.round(sb_index[1] / len_fe_frames)))] = c + 1
                    sb_track[range(int(np.floor(sb_index[0] / len_fe_frames)),
                                   int(np.floor(sb_index[1] / len_fe_frames)))] = c + 1
                sb_track[-1] = sb_track[-2]
            else:
                algo = rpt.Pelt(model="rbf").fit(features)
                result = algo.predict(pen=0.63)
                sb_track = np.zeros(len(features))
                sb_track[:result[0]]=1
                for c,cpi in enumerate(result[1:]):
                    sb_track[result[c]:result[c+1]]=c+2



            ind_track = np.arange(len(features))

            if full_Len:
                self.list_features.append(features)
                self.list_features_sb_track.append(torch.tensor(sb_track))
                self.list_features_ind_track.append(torch.tensor(ind_track))
            elif dw_s == 1:
                features = pad(features, (0, 0, 0, Tr_seq_Len - len(features) % Tr_seq_Len), mode='constant', value=0)
                sb_track = pad(torch.tensor(sb_track), (0, Tr_seq_Len - len(sb_track) % Tr_seq_Len), mode='constant',
                               value=0)
                ind_track = pad(torch.tensor(ind_track), (0, Tr_seq_Len - len(ind_track) % Tr_seq_Len), mode='constant',
                               value=-1)
                # features = features[:len(features)- len(features) % Tr_seq_Len,:]
                # sb_track = torch.tensor(sb_track)[:len(sb_track)- len(sb_track) % Tr_seq_Len]

                features = features.view([Tr_seq_Len, -1, features.size(-1)]).permute(1, 0, 2)
                sb_track = sb_track.view([Tr_seq_Len, -1]).permute(1, 0)
                ind_track = ind_track.view([Tr_seq_Len, -1]).permute(1, 0)
                self.list_features.extend(torch.tensor(x) for x in features.tolist())
                self.list_features_sb_track.extend(torch.tensor(x) for x in sb_track.tolist())
                self.list_features_ind_track.extend(torch.tensor(x) for x in ind_track.tolist())
            elif dw_s == 2:
                # features = pad(features, (0, 0, 0, Tr_seq_Len - len(features) % Tr_seq_Len), mode='constant', value=0)
                # sb_track = pad(torch.tensor(sb_track), (0, Tr_seq_Len - len(sb_track) % Tr_seq_Len), mode='constant',
                #                value=0)
                features = features[:len(features) - len(features) % Tr_seq_Len, :]
                sb_track = torch.tensor(sb_track)[:len(sb_track) - len(sb_track) % Tr_seq_Len]
                ind_track = torch.tensor(ind_track)[:len(ind_track) - len(ind_track) % Tr_seq_Len]

                features = features.view([Tr_seq_Len, -1, features.size(-1)]).permute(1, 0, 2)
                sb_track = sb_track.view([Tr_seq_Len, -1]).permute(1, 0)
                ind_track = ind_track.view([Tr_seq_Len, -1]).permute(1, 0)
                self.list_features.extend(torch.tensor(x) for x in features.tolist())
                self.list_features_sb_track.extend(torch.tensor(x) for x in sb_track.tolist())
                self.list_features_ind_track.extend(torch.tensor(x) for x in ind_track.tolist())
            elif dw_s == 3:

                if overlap:
                    if len(features) > Tr_seq_Len:
                        stide = int((-Tr_seq_Len + len(features)) / round(len(features) / Tr_seq_Len))
                        self.list_features.extend(
                            torch.tensor(x).permute(1, 0) for x in features.unfold(0, Tr_seq_Len, int(stide)).tolist())
                        self.list_features_sb_track.extend(
                            torch.tensor(x) for x in torch.tensor(sb_track).unfold(0, Tr_seq_Len, int(stide)).tolist())
                        self.list_features_ind_track.extend(
                            torch.tensor(x) for x in torch.tensor(ind_track).unfold(0, Tr_seq_Len, int(stide)).tolist())
                    else:
                        self.list_features.append(pad(features, (0, 0, 0, Tr_seq_Len - len(features)), mode='constant', value=0))
                        self.list_features_sb_track.append(pad(torch.tensor(sb_track), (0, Tr_seq_Len - len(sb_track)), mode='constant',
                               value=0))
                        self.list_features_ind_track.append(pad(torch.tensor(ind_track), (0, Tr_seq_Len - len(ind_track)), mode='constant',
                                value=-1))
                else:
                    self.list_features.extend(torch.split(features, Tr_seq_Len))
                    self.list_features_sb_track.extend(torch.split(torch.tensor(sb_track), Tr_seq_Len))
                    self.list_features_ind_track.extend(torch.split(torch.tensor(ind_track), Tr_seq_Len))
                features = pad(features, (0, 0, 0, Tr_seq_Len - len(features) % Tr_seq_Len), mode='constant', value=0)
                sb_track = pad(torch.tensor(sb_track), (0, Tr_seq_Len - len(sb_track) % Tr_seq_Len), mode='constant',
                               value=0)
                ind_track = pad(torch.tensor(ind_track), (0, Tr_seq_Len - len(ind_track) % Tr_seq_Len), mode='constant',
                                value=-1)
                # features = features[:len(features)- len(features) % Tr_seq_Len,:]
                # sb_track = torch.tensor(sb_track)[:len(sb_track)- len(sb_track) % Tr_seq_Len]

                features = features.view([Tr_seq_Len, -1, features.size(-1)]).permute(1, 0, 2)
                sb_track = sb_track.view([Tr_seq_Len, -1]).permute(1, 0)
                ind_track = ind_track.view([Tr_seq_Len, -1]).permute(1, 0)
                self.list_features.extend(torch.tensor(x) for x in features.tolist())
                self.list_features_sb_track.extend(torch.tensor(x) for x in sb_track.tolist())
                self.list_features_ind_track.extend(torch.tensor(x) for x in ind_track.tolist())
            elif dw_s == 7:

                if overlap:
                    if len(features) > Tr_seq_Len:
                        stide = 3
                        self.list_features.extend(
                            torch.tensor(x).permute(1, 0) for x in features.unfold(0, Tr_seq_Len, int(stide)).tolist())
                        self.list_features_sb_track.extend(
                            torch.tensor(x) for x in torch.tensor(sb_track).unfold(0, Tr_seq_Len, int(stide)).tolist())
                        self.list_features_ind_track.extend(
                            torch.tensor(x) for x in torch.tensor(ind_track).unfold(0, Tr_seq_Len, int(stide)).tolist())
                    # else:
                    #     self.list_features.append(features)
                    #     self.list_features_sb_track.append(torch.tensor(sb_track))
                    #     self.list_features_ind_track.append(torch.tensor(ind_track))
                else:
                    self.list_features.extend(torch.split(features, Tr_seq_Len))
                    self.list_features_sb_track.extend(torch.split(torch.tensor(sb_track), Tr_seq_Len))
                    self.list_features_ind_track.extend(torch.split(torch.tensor(ind_track), Tr_seq_Len))
                features = pad(features, (0, 0, 0, Tr_seq_Len - len(features) % Tr_seq_Len), mode='constant', value=0)
                sb_track = pad(torch.tensor(sb_track), (0, Tr_seq_Len - len(sb_track) % Tr_seq_Len), mode='constant',
                               value=0)
                ind_track = pad(torch.tensor(ind_track), (0, Tr_seq_Len - len(ind_track) % Tr_seq_Len), mode='constant',
                                value=-1)
                # features = features[:len(features)- len(features) % Tr_seq_Len,:]
                # sb_track = torch.tensor(sb_track)[:len(sb_track)- len(sb_track) % Tr_seq_Len]

                features = features.view([Tr_seq_Len, -1, features.size(-1)]).permute(1, 0, 2)
                sb_track = sb_track.view([Tr_seq_Len, -1]).permute(1, 0)
                ind_track = ind_track.view([Tr_seq_Len, -1]).permute(1, 0)
                self.list_features.extend(torch.tensor(x) for x in features.tolist())
                self.list_features_sb_track.extend(torch.tensor(x) for x in sb_track.tolist())
                self.list_features_ind_track.extend(torch.tensor(x) for x in ind_track.tolist())
            elif dw_s == 4 or dw_s == 5 or dw_s==6:


                if len(features) > (Tr_seq_Len*1.5):
                    stide = int((-(Tr_seq_Len*1.5) + len(features)) / round(len(features) / (Tr_seq_Len*1.5)))
                    self.list_features.extend(
                        torch.tensor(x).permute(1, 0) for x in features.unfold(0, int(Tr_seq_Len*1.5), int(stide)).tolist())
                    self.list_features_sb_track.extend(
                        torch.tensor(x) for x in torch.tensor(sb_track).unfold(0, int(Tr_seq_Len*1.5), int(stide)).tolist())
                    self.list_features_ind_track.extend(
                        torch.tensor(x) for x in torch.tensor(ind_track).unfold(0, int(Tr_seq_Len*1.5), int(stide)).tolist())
                # else:
                #     self.list_features.append(features)
                #     self.list_features_sb_track.append(torch.tensor(sb_track))
                #     self.list_features_ind_track.append(torch.tensor(ind_track))

                features = pad(features, (0, 0, 0, Tr_seq_Len - len(features) % Tr_seq_Len), mode='constant', value=0)
                sb_track = pad(torch.tensor(sb_track), (0, Tr_seq_Len - len(sb_track) % Tr_seq_Len), mode='constant',
                               value=0)
                ind_track = pad(torch.tensor(ind_track), (0, Tr_seq_Len - len(ind_track) % Tr_seq_Len), mode='constant',
                                value=-1)
                self.list_features.append(features)
                self.list_features_sb_track.append(sb_track)
                self.list_features_ind_track.append(ind_track)
            elif dw_s ==8:


                if len(features) > (Tr_seq_Len*1.5):
                    stide = int((-(Tr_seq_Len*0.5) + len(features)) / round(len(features) / (Tr_seq_Len*0.5)))
                    self.list_features.extend(
                        torch.tensor(x).permute(1, 0) for x in features.unfold(0, int(Tr_seq_Len*1.5), int(stide)).tolist())
                    self.list_features_sb_track.extend(
                        torch.tensor(x) for x in torch.tensor(sb_track).unfold(0, int(Tr_seq_Len*1.5), int(stide)).tolist())
                    self.list_features_ind_track.extend(
                        torch.tensor(x) for x in torch.tensor(ind_track).unfold(0, int(Tr_seq_Len*1.5), int(stide)).tolist())
                # else:
                #     self.list_features.append(features)
                #     self.list_features_sb_track.append(torch.tensor(sb_track))
                #     self.list_features_ind_track.append(torch.tensor(ind_track))

                features = pad(features, (0, 0, 0, Tr_seq_Len - len(features) % Tr_seq_Len), mode='constant', value=0)
                sb_track = pad(torch.tensor(sb_track), (0, Tr_seq_Len - len(sb_track) % Tr_seq_Len), mode='constant',
                               value=0)
                ind_track = pad(torch.tensor(ind_track), (0, Tr_seq_Len - len(ind_track) % Tr_seq_Len), mode='constant',
                                value=-1)
                features=features.view([Tr_seq_Len, -1, features.size(-1)]).permute(1, 0, 2)
                sb_track = sb_track.view([Tr_seq_Len, -1]).permute(1, 0)
                ind_track = ind_track.view([Tr_seq_Len, -1]).permute(1, 0)
                self.list_features.extend(torch.tensor(x) for x in features.tolist())
                self.list_features_sb_track.extend(torch.tensor(x) for x in sb_track.tolist())
                self.list_features_ind_track.extend(torch.tensor(x) for x in ind_track.tolist())
                # features = features.view([Tr_seq_Len, -1, features.size(-1)]).permute(1, 0, 2)
                # sb_track = sb_track.view([Tr_seq_Len, -1]).permute(1, 0)
                # ind_track = ind_track.view([Tr_seq_Len, -1]).permute(1, 0)
                # self.list_features.extend(torch.tensor(x) for x in features.tolist())
                # self.list_features_sb_track.extend(torch.tensor(x) for x in sb_track.tolist())
                # self.list_features_ind_track.extend(torch.tensor(x) for x in ind_track.tolist())

            else:
                # self.list_features.extend(torch.split(features,Tr_seq_Len))
                # stide=Tr_seq_Len-np.ceil((Tr_seq_Len-(len(features)%Tr_seq_Len))/2) -1
                
                if overlap:
                    if len(features) > Tr_seq_Len:
                        stide = int((-Tr_seq_Len + len(features)) / round(len(features) / Tr_seq_Len))
                        self.list_features.extend(
                            torch.tensor(x).permute(1, 0) for x in features.unfold(0, Tr_seq_Len, int(stide)).tolist())
                        self.list_features_sb_track.extend(
                            torch.tensor(x) for x in torch.tensor(sb_track).unfold(0, Tr_seq_Len, int(stide)).tolist())
                        self.list_features_ind_track.extend(
                            torch.tensor(x) for x in torch.tensor(ind_track).unfold(0, Tr_seq_Len, int(stide)).tolist())
                    else:
                        self.list_features.append(features)
                        self.list_features_sb_track.append(torch.tensor(sb_track))
                        self.list_features_ind_track.append(torch.tensor(ind_track))
                else:
                    self.list_features.extend(torch.split(features, Tr_seq_Len))
                    self.list_features_sb_track.extend(torch.split(torch.tensor(sb_track), Tr_seq_Len))
                    self.list_features_ind_track.extend(torch.split(torch.tensor(ind_track), Tr_seq_Len))

            len2 = len(self.list_features)
            self.list_vid_ind.extend(video_name for i in range(len2-len1))

            # Max_len= len(features) if len(features)>Max_len else Max_len
            # self.action_fragments[video_name] = compute_fragments(features.shape[0], action_state_size)

        hdf.close()

        # ppad
        #self.pad_feartures = pad_sequence(self.list_features, batch_first=True)
        #self.pad_sb = pad_sequence(self.list_features_sb_track, batch_first=True)
        #self.pad_ind = pad_sequence(self.list_features_ind_track, batch_first=True)
        self.pad_feartures = self.list_features
        self.pad_sb = self.list_features_sb_track
        self.pad_ind = self.list_features_ind_track
        
        # if self.mode == 'train':
        #if Records['Max_Len']<self.pad_feartures.size(1):

         #   Records['Max_Len'] =self.pad_feartures.size(1)
        #     Records['Max_Len'] = self.pad_feartures.size(1)
        # else:
        #     self.pad_feartures = pad(self.pad_feartures,
        #                              (0, 0, 0, Records['Max_Len'] - self.pad_feartures.size(1), 0, 0), mode='constant',
        #                              value=0)
        #     self.pad_sb = pad(self.pad_sb, (0, 0, 0, Records['Max_Len'] - self.pad_feartures.size(1)),
        #                       mode='constant', value=0)
        # self.maskRand( self.pad_feartures[13], self.pad_sb[13])

        if self.mode == 'test':
            if self.Masking:
                if self.DlTyp==0:
                    self.pad_feartures2, self.pad_sb2, self.orig_features, self.masked_indicate, self.list_vid_ind2, self.list_mask_ind2 = self.maskRandTest(self.pad_feartures.copy(), self.pad_sb.copy(), self.list_vid_ind, self.pad_ind.copy())
                elif self.DlTyp==1:
                    self.pad_feartures2, self.pad_sb2, self.orig_features, self.masked_indicate, self.list_vid_ind2, self.list_mask_ind2 = self.maskFullshotTest(
                        self.pad_feartures.copy(), self.pad_sb.copy(), self.list_vid_ind.copy(), self.pad_ind.copy())
                elif self.DlTyp == 2:
                    self.pad_feartures2, self.pad_sb2, self.orig_features, self.masked_indicate, self.list_vid_ind2, self.list_mask_ind2 = self.maskCPTest(
                        self.pad_feartures.copy(), self.pad_sb.copy(), self.list_vid_ind.copy(), self.pad_ind.copy())
                elif self.DlTyp == 3:
                    self.pad_feartures2, self.pad_sb2, self.orig_features, self.masked_indicate, self.list_vid_ind2, self.list_mask_ind2 = self.maskRandShotTest(
                        self.pad_feartures.copy(), self.pad_sb.copy(), self.list_vid_ind.copy(), self.pad_ind.copy())

    def Random_Data_augment(self,features, sb, pad_ind):
        random.seed(random.SystemRandom().randint(0, sys.maxsize))
        np.random.seed(random.SystemRandom().randint(0, 2**32 -1))
        if len(features) <= self.Tr_seq_Len:
            features_o = pad(features, (0, 0, 0, self.Tr_seq_Len - len(features)), mode='constant', value=0)
            sb_o = pad(sb, (0, self.Tr_seq_Len - len(features)), mode='constant',
                           value=0)
            pad_ind_o = pad(pad_ind, (0, self.Tr_seq_Len - len(features)), mode='constant',
                            value=-1)
        elif len(features) == self.Tr_seq_Len*1.5:
            sb_bt = sb[int(self.Tr_seq_Len / 4)]
            sb_tp = sb[int(self.Tr_seq_Len * 3 / 4)]
            bt_sbb_bt=(self.Tr_seq_Len / 4) - min((sb==sb_bt).nonzero())
            bt_sbb_tp=max((sb==sb_bt).nonzero()) - (self.Tr_seq_Len / 4)
            tp_sbb_bt = (self.Tr_seq_Len * 3 / 4) - min((sb == sb_tp).nonzero())
            tp_sbb_tp = max((sb == sb_tp).nonzero()) - (self.Tr_seq_Len * 3 / 4)
            if self.dw_s==4:
                rg=min([bt_sbb_tp,bt_sbb_bt,tp_sbb_bt,tp_sbb_tp])
                rg=min([rg,int(self.Tr_seq_Len / 4)])
            elif self.dw_s==5:
                rg = max([bt_sbb_tp, bt_sbb_bt, tp_sbb_bt, tp_sbb_tp])
                rg = min([rg, int(self.Tr_seq_Len / 4)])
            elif self.dw_s == 6 or self.dw_s == 8:

                rg = int(self.Tr_seq_Len / 4)

            ind_s=random.randint(self.Tr_seq_Len / 4 - rg, self.Tr_seq_Len / 4 + rg)
            features_o=features[ind_s:ind_s+self.Tr_seq_Len,:]
            sb_o=sb[ind_s:ind_s+self.Tr_seq_Len]
            pad_ind_o=pad_ind[ind_s:ind_s+self.Tr_seq_Len]
        else:
            features = features.view([self.Tr_seq_Len, -1, features.size(-1)])
            sb = sb.view([self.Tr_seq_Len, -1])
            pad_ind = pad_ind.view([self.Tr_seq_Len, -1])


            ind_list=np.random.random_integers(0, features.size(1)-1, self.Tr_seq_Len)
            features_o=torch.stack([f[ind_list[c],:] for c,f in enumerate(features)])
            pad_ind_o=torch.stack([f[ind_list[c]] for c,f in enumerate(pad_ind)])
            sb_o = torch.stack([f[ind_list[c]] for c,f in enumerate(sb)])

            # features = features.view([self.Tr_seq_Len, -1, features.size(-1)]).permute(1, 0, 2)
            # sb = sb.view([self.Tr_seq_Len, -1]).permute(1, 0)
            # pad_ind = pad_ind.view([self.Tr_seq_Len, -1]).permute(1, 0)
            # idx=np.random.random_integers(0, len(features)-1,1)
            # features_o = features[idx,:,:].squeeze(0)
            # pad_ind_o=pad_ind[idx,:].squeeze(0)
            # sb_o = sb[idx,:].squeeze(0)


        return features_o,sb_o,pad_ind_o
    def maskRandShotTest(self, features, sb, vid_ind, pad_ind, seedm=111):

        out_features = []
        orig_features = []
        out_sb = []
        out_mask = []
        out_mask_ind = []
        out_vid_index = []
        torch.manual_seed(seedm)
        for m_r in self.mask_ratio:
            for c, fe in enumerate(features):
                fe, sb[c], pad_ind[c] = self.Random_Data_augment(fe, sb[c], pad_ind[c])
                sb_idx=torch.randperm(len(sb[c].unique()))
                sb_prm=sb[c].unique()[sb_idx]
                mask = torch.ones_like(sb[c])
                mask[sb[c] == 0] = 0
                max_masked = np.ceil(m_r * mask.sum())
                mask = torch.zeros_like(sb[c])
                for sb_el in sb_prm:
                    if sb_el!=0:
                        mask[sb[c] == sb_el] = 1
                        if mask.sum() >= max_masked or sb_el==sb_prm[-1]:
                            zero_ind=(mask==1).nonzero(as_tuple=True)[0].tolist()
                            orig_features.append(fe)
                            out_mask.append(mask)
                            out_sb.append(sb[c])
                            out_features.append(fe.masked_fill(mask.unsqueeze(-1).expand_as(fe) == 1, 1e-9))
                            out_vid_index.append(vid_ind[c])

                            if self.invert_loss:
                                out_mask_ind.append(pad_ind[c][zero_ind])
                            else:
                                out_mask_ind.append(pad_ind[c][[x for x in range(len(pad_ind[c])) if
                                                                (x not in zero_ind) and pad_ind[c][x] != (-1)]])
                            mask = torch.zeros_like(sb[c])
        out_mask_ind = list(zip(*itertools.zip_longest(*out_mask_ind, fillvalue=-1)))
        return out_features, out_sb, orig_features, out_mask, out_vid_index, out_mask_ind
    def maskFullshotTest(self, features, sb, vid_ind, pad_ind, seedm=111):
        out_features = []
        orig_features = []
        out_sb = []
        out_mask = []
        out_mask_ind = []
        out_vid_index = []
        # torch.manual_seed(seedm)
        features=torch.stack(features)
        sb=torch.stack(sb)
        pad_ind=torch.stack(pad_ind)
        for sht_idx in sb.unique():
            if sht_idx !=0:
                mask_idx=(sb == sht_idx).nonzero(as_tuple=False)
                mask=(sb == sht_idx).unsqueeze(-1).repeat(1,1,features.size(2))
                lst_ad=list(features.masked_fill(mask == 1, 1e-9))
                for i in range(len(sb)):
                    if i not in mask_idx[:,0].tolist():
                        lst_ad[i]=[]
                lst_ad = [value for value in lst_ad if value != []]
                out_features.extend(lst_ad)

                lst_ad = list(features)
                for i in range(len(sb)):
                    if i not in mask_idx[:, 0].tolist():
                        lst_ad[i] = []
                lst_ad = [value for value in lst_ad if value != []]
                orig_features.extend(lst_ad)

                lst_ad = list(sb)
                for i in range(len(sb)):
                    if i not in mask_idx[:, 0].tolist():
                        lst_ad[i] = []
                lst_ad = [value for value in lst_ad if value != []]
                out_sb.extend(lst_ad)

                lst_ad = list((sb == sht_idx).long())
                for i in range(len(sb)):
                    if i not in mask_idx[:, 0].tolist():
                        lst_ad[i] = []
                lst_ad = [value for value in lst_ad if value != []]
                out_mask.extend(lst_ad)
                if self.invert_loss:
                    lst_ad = list(pad_ind.masked_fill(sb != sht_idx, -1))
                else:
                    lst_ad = list(pad_ind.masked_fill(sb == sht_idx, -1))
                for i in range(len(sb)):
                    if i not in mask_idx[:, 0].tolist():
                        lst_ad[i] = []
                lst_ad = [value for value in lst_ad if value != []]
                out_mask_ind.extend(lst_ad)
                # for vidx in mask_idx[:,0].unique():
                #     out_vid_index.append(vid_ind[vidx])
                lst_ad = vid_ind.copy()
                for i in range(len(sb)):
                    if i not in mask_idx[:, 0].tolist():
                        lst_ad[i] = []
                lst_ad = [value for value in lst_ad if value != []]
                out_vid_index.extend(lst_ad)

        return out_features, out_sb, orig_features, out_mask, out_vid_index, out_mask_ind

    def maskCPTest(self, features, sb_i, vid_ind, pad_ind_i, seedm=111):

        out_features = []
        orig_features = []
        out_sb = []
        out_mask = []
        out_mask_ind = []
        sb = sb_i[:]
        pad_ind = pad_ind_i[:]
        out_vid_index = []
        torch.manual_seed(seedm)
        c = list(itertools.product(self.mask_ratio, self.window_s))
        for mask_ratio, window_s in c:
            for c, fe in enumerate(features):
                fe, sb[c], pad_ind[c] = self.Random_Data_augment(fe, sb_i[c], pad_ind_i[c])

                fe1=torch.roll(fe,1,0)
                fe2 = torch.roll(fe, -1, 0)
                difsumf=torch.mean(torch.abs(fe-fe1)+torch.abs(fe-fe2),1)

                mask = torch.ones_like(sb[c], dtype=bool)
                mask[sb[c] == 0] = 0
                max_masked = np.ceil(mask_ratio * mask.sum() / window_s)
                _, indices_ord = torch.topk(difsumf.masked_fill(~mask, 1e-9), len(difsumf))
                maske_ind_list = []
                masked_ind_ra = []
                for ind in indices_ord.tolist():
                    if ind not in masked_ind_ra:
                        if self.Full_shot_mask:
                            sb_r = (sb[c] == sb[c][ind]).nonzero().flatten().tolist()
                            sb_l = min(sb_r)
                            sb_h = max(sb_r)
                            sb_hmax = (sb_l + window_s) if sb_l > (ind - round(window_s / 2)) else (
                                    ind + round(window_s / 2))
                            sb_Lmin = (sb_h - window_s) if sb_h < (ind + round(window_s / 2)) else (
                                    ind - round(window_s / 2))
                            maske_ind_list.append([max(sb_l, sb_Lmin), min(sb_h, sb_hmax) + 1])
                            # maske_ind_list.append(
                            #     [sb_l,sb_h])
                            masked_ind_ra.extend(range(maske_ind_list[-1][0], maske_ind_list[-1][1]))
                            max_masked -= (maske_ind_list[-1][1] - maske_ind_list[-1][0])
                        else:
                            maske_ind_list.append(
                                [ind - round(self.window_s / 2), ind + round(self.window_s / 2) + 1])
                            max_masked -= (maske_ind_list[-1][1] - maske_ind_list[-1][0])
                            masked_ind_ra.extend(range(maske_ind_list[-1][0], maske_ind_list[-1][1]))
                    # else:
                    #     print(ind)
                    if max_masked <= 0:
                        break
                mask = torch.zeros_like(fe)
                mask[masked_ind_ra, :] = 1
                orig_features.append(fe)
                out_features.append(fe.masked_fill(mask == 1, 1e-9))
                out_mask.append(mask[:, 0].squeeze(-1))
                if self.invert_loss:
                    out_mask_ind.append(pad_ind[c][masked_ind_ra])
                else:
                    out_mask_ind.append(pad_ind[c][[x for x in range(len(pad_ind[c])) if
                                                    (x not in masked_ind_ra) and pad_ind[c][x] != (-1)]])
                out_sb.append(sb[c])
                out_vid_index.extend(vid_ind[c])

        out_mask_ind = list(zip(*itertools.zip_longest(*out_mask_ind, fillvalue=-1)))
        return out_features, out_sb, orig_features, out_mask, out_vid_index, out_mask_ind

    def maskRandTest(self, features, sb_i,vid_ind,pad_ind_i,seedm=111):

        out_features = []
        orig_features = []
        out_sb = []
        out_mask = []
        out_mask_ind = []
        out_vid_index=[]
        torch.manual_seed(seedm)
        sb=sb_i[:]
        pad_ind=pad_ind_i[:]
        c=list(itertools.product(self.mask_ratio,self.window_s))
        for mask_ratio, window_s in c:
            for c, fe in enumerate(features):
                fe, sb[c], pad_ind[c] = self.Random_Data_augment(fe, sb_i[c], pad_ind_i[c])
                len1=len(out_features)
                mask = torch.ones_like(sb[c], dtype=bool)
                mask[sb[c] == 0] = 0
                if mask_ratio>0.5:
                    max_masked = np.ceil((1-mask_ratio) * mask.sum() / window_s)
                # add seeed
                    rand_ind_o = torch.randperm(int(np.ceil(mask.sum() / window_s)))
                    rand_ind_t = rand_ind_o.unfold(0, int(max_masked), int(max_masked)).tolist()
                    rand_ind=[]
                    for rnd_ind_list in rand_ind_t:
                        r_temp=[elt for elt in rand_ind_o if elt not in rnd_ind_list]
                        rand_ind.append(r_temp)

                else:
                    max_masked = np.ceil(mask_ratio * mask.sum() / window_s)
                    rand_ind = torch.randperm(int(np.ceil(mask.sum() / window_s)))
                    rand_ind = rand_ind.unfold(0, int(max_masked), int(max_masked)).tolist()
                for elm in rand_ind:
                    mask = torch.zeros_like(fe)

                    zero_ind = list(
                        itertools.chain(*(map(lambda x: list(range(x * window_s, (x + 1) * window_s)), elm))))
                    mask[zero_ind, :] = 1
                    orig_features.append(fe)
                    out_features.append(fe.masked_fill(mask == 1, 1e-9))
                    out_mask.append(mask[:, 0].squeeze(-1))
                    # out_mask_ind.append(pad_ind[c][zero_ind])
                    if self.invert_loss:
                        out_mask_ind.append(pad_ind[c][zero_ind])
                    else:
                        out_mask_ind.append(pad_ind[c][[x for x in range(len(pad_ind[c])) if
                                                        (x not in zero_ind) and pad_ind[c][x] != (-1)]])
                    out_sb.append(sb[c])
                len2 = len(out_features)
                out_vid_index.extend(vid_ind[c] for i in range(len2-len1))
            # mask_zero_ind = [item for sublist in mask_zero_ind for item in mask_zero_ind]
            # out_features.extend(torch.tensor(x).permute(1, 0) for x in features.unfold(0,1, int(1)).tolist())
        out_mask_ind = list(zip(*itertools.zip_longest(*out_mask_ind, fillvalue=-1)))
        return out_features, out_sb, orig_features, out_mask,out_vid_index,out_mask_ind

    def maskRand(self, features, sb):
        mask = torch.ones_like(sb, dtype=bool)

        mask[sb == 0] = 0
        torch.manual_seed(self.rng.randint(0,sys.maxsize))
        max_masked = np.ceil(self.mask_ratio * mask.sum())
        _, indices_ord = torch.topk(torch.rand(len(mask)).masked_fill(~mask, 1e-9), len(mask))
        maske_ind_list = []
        masked_ind_ra = []

        for ind in indices_ord.tolist():
            if ind not in masked_ind_ra:
                if self.Full_shot_mask:
                    sb_r = (sb == sb[ind]).nonzero().flatten().tolist()
                    sb_l = min(sb_r)
                    sb_h = max(sb_r)
                    sb_hmax = (sb_l + self.window_s) if sb_l > (ind - round(self.window_s / 2)) else (
                            ind + round(self.window_s / 2))
                    sb_Lmin = (sb_h - self.window_s) if sb_h < (ind + round(self.window_s / 2)) else (
                            ind - round(self.window_s / 2))
                    maske_ind_list.append([max(sb_l, sb_Lmin), min(sb_h, sb_hmax)+1])
                    # maske_ind_list.append(
                    #     [sb_l,sb_h])
                    masked_ind_ra.extend(range(maske_ind_list[-1][0], maske_ind_list[-1][1]))
                    max_masked -= (maske_ind_list[-1][1] - maske_ind_list[-1][0])
                else:
                    maske_ind_list.append(
                        [ind - round(self.window_s / 2), ind + round(self.window_s / 2)+1])
                    max_masked -= (maske_ind_list[-1][1] - maske_ind_list[-1][0])
                    masked_ind_ra.extend(range(maske_ind_list[-1][0], maske_ind_list[-1][1]))
            # else:
            #     print(ind)
            if max_masked <= 0:
                break
        mask_filter = torch.rand(len(maske_ind_list))
        indices_to_zero = []
        all_indices = list(range(mask.sum()))
        # all_indices = [x for x in all_indices if x not in masked_ind_ra]
        mask_out = torch.zeros(features.size(0))
        for c, ms in enumerate(maske_ind_list):
            if mask_filter[c] < self.mask_chance:
                indices_to_zero.extend(range(ms[0], ms[1]))
                mask_out[range(ms[0], ms[1])] = 1
            elif mask_filter[c] < self.mask_chance + self.replace_chance:
                replace_ok_ind = [(x - ms[1] + ms[0]) for x in all_indices if x not in list(range(ms[0], ms[1]))]
                replace_ok_ind = [x for x in replace_ok_ind if x >= 0]
                random_num = random.choice(replace_ok_ind)
                features[range(ms[0], ms[1]), :] = features[range(random_num, random_num + ms[1] - ms[0]), :]
                mask_out[range(ms[0], ms[1])] = 2
            else:
                mask_out[range(ms[0], ms[1])] = 3
        features[indices_to_zero, :] = 1e-9
        return features, mask_out

    def __len__(self):
        self.len = len(self.pad_feartures) if self.mode == 'train' else len(self.pad_feartures2)
        return self.len

    # In "train" mode it returns the features and the action_fragments; in "test" mode it also returns the video_name
    def __getitem__(self, index):
        vid_ind = []
        mask_ind = []
        if self.Masking:
            if self.mode == 'train':


                features = self.pad_feartures[index].clone()
                sb = self.pad_sb[index].clone()
                ind_track=self.list_features_ind_track[index].clone()
                features,sb,ind_track=self.Random_Data_augment(features,sb,ind_track)
                attention_mask = (sb != 0)
                masked_fe, masked_indicate = self.maskRand(features.clone(), sb)
                # features = self.pad_feartures[index]
                pass
            else:
                masked_fe = self.pad_feartures2[index]
                sb = self.pad_sb2[index]
                attention_mask = (sb != 0)
                features = self.orig_features[index]
                masked_indicate = self.masked_indicate[index]
                vid_ind = self.list_vid_ind2[index]
                mask_ind = self.list_mask_ind2[index]
        else:
            masked_fe = self.pad_feartures[index]
            features = self.pad_feartures[index].clone()
            sb = self.pad_sb[index]
            attention_mask = (sb != 0)
            masked_indicate = torch.zeros_like(sb)
        return masked_fe, attention_mask, features, masked_indicate, vid_ind, torch.tensor(mask_ind)

    def update(self, mseed):
        if self.DlTyp==0:
            self.pad_feartures2, self.pad_sb2, self.orig_features, self.masked_indicate, self.list_vid_ind2, self.list_mask_ind2 = self.maskRandTest(
                self.pad_feartures.copy(), self.pad_sb.copy(), self.list_vid_ind.copy(), self.pad_ind.copy(), mseed)
        elif self.DlTyp==3:
            self.pad_feartures2, self.pad_sb2, self.orig_features, self.masked_indicate, self.list_vid_ind2, self.list_mask_ind2 = self.maskRandShotTest(
                self.pad_feartures.copy(), self.pad_sb.copy(), self.list_vid_ind.copy(), self.pad_ind.copy())
        else:
            pass
if __name__ == '__main__':
    # train_loader = VideoData('train', 0)
    # train_loader = VideoData('test', 0)
    # a, b, c = next(iter(train_loader))
    pass
