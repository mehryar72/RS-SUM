import sys
from os import listdir
import json
import numpy as np
import h5py
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary
import csv
from configs import get_config
from solver import Solver
from data_loader import VideoData, Records
import torch
import logging

def reslog(line):
    with open(r'/home/mabbasib/Transformer_pretr/results/acc2.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(line)
        f.close()
if __name__ == '__main__':
    config = get_config()
    logF = config.name + '__.csv'
    # tb_writer = SummaryWriter(config.name2)
    print(config)
    # if (os.path.isfile(logF)):
        # os.remove(logF)
    logging.basicConfig(filename=logF,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger('myloger')
    #trainset = VideoData('train', config.split_index,
    #                     name='summe' if config.video_type == 'SumMe' else 'tvsum',
    #                     dw_s=config.dw_s, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
    #                     Masking=config.Masking,
    #                     Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio,
    #                     mask_chance=config.mask_chance, replace_chance=config.replace_chance, overlap=config.overlap,
    #                     window_s=config.window_s)
    #trainset1 = VideoData('train', config.split_index,
    #                      name='tvsum' if config.video_type == 'SumMe' else 'summe',
    #                      dw_s=config.dw_s, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
    #                      Masking=config.Masking,
    #                      Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio,
    #                      mask_chance=config.mask_chance, replace_chance=config.replace_chance,
    #                      overlap=config.overlap,
    #                      window_s=config.window_s)
    #trainset2 = VideoData('train', config.split_index,
    #                      name='tvsum' if config.video_type == 'SumMe' else 'summe',
    #                      dw_s=config.dw_s, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
    #                      Masking=config.Masking,
    #                      Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio,
    #                      mask_chance=config.mask_chance, replace_chance=config.replace_chance,
    #                      overlap=config.overlap,
    #                      window_s=config.window_s,
    #                      CombD=1)
    #trainset = torch.utils.data.ConcatDataset([trainset, trainset1, trainset2])

    #testset = VideoData('test', config.split_index,
    #                    name='summe' if config.video_type == 'SumMe' else 'tvsum',
    #                    dw_s=config.dw_s_te, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
    #                    Masking=config.Masking,
    #                    Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio_te,
    #                    mask_chance=config.mask_chance, replace_chance=config.replace_chance, overlap=config.overlap,
    #                    window_s=config.window_s_te)
    testset2 = VideoData('test', config.split_index,
                         name='summe' if config.video_type == 'SumMe' else 'tvsum',
                         dw_s=config.dw_s_fs, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                         Masking=config.Masking,
                         Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio_fs,
                         mask_chance=config.mask_chance, replace_chance=config.replace_chance, overlap=config.overlap,
                         window_s=config.window_s_fs,
                         DlTyp=config.DlTyp_fs,
                         invert_loss=config.invert_loss)
    #trainLoader = torch.utils.data.DataLoader(trainset,
    #                                          batch_size=config.batch_size, shuffle=True, num_workers=0)
    #testLoader = torch.utils.data.DataLoader(testset,
    #                                         batch_size=config.batch_size, shuffle=False, num_workers=0)
    dataset_lens = {'train': len(testset2), 'test': len(testset2)}
    solver = Solver(config, None, None, Records['Max_Len'], logger, dataset_lens, testset2)
    solver.build()
    if config.DlTyp_fs==0 or config.DlTyp_fs==3:
        loss_dic = solver.score(100)
    else:
        loss_dic = solver.score(1)

    DATASET_PATH= '../data/' + config.video_type + '/eccv16_dataset_' + config.video_type.lower() + '_google_pool5.h5'

    all_f_scores_elm_dict={'L':[],'R':[],'RL':[]}
    ScoresLR={'L':[],'R':[]}
    hdf= h5py.File(DATASET_PATH, 'r')
    sclist = ['R', 'L', 'Rs-L_0.1', 'Rs-L_0.2', 'Rs-L_0.3', 'Rs-L_0.4', 'Rs-L_0.5', 'Rs-L_0.6', 'Rs-L_0.7', 'Rs-L_0.8',
              'Rs-L_0.9', 'Rs-L_1', 's-L'] if not config.invert_loss else ['R', 'L', 's-L', 's-L', 'sL_1',
                                                                           'RsL_0.2', 'Rs-L_0.2', 'RsL_0.5', 'Rs-L_0.5', 'RsL_0.8', 'Rs-L_0.8', 'RsL_1', 'Rs-L_1']
    print(sclist)
    for elmpath in sclist:
    #for elmpath in ['R','L','sL','R-L','Rs-L_1','s-L']:
        if elmpath=='RL':
            all_scores = [a + b for a, b in zip(ScoresLR['L'][0], ScoresLR['R'][0])]
            all_scores2 = [a + b for a, b in zip(ScoresLR['L'][1], ScoresLR['R'][1])]
            all_scores3 = [a + b for a, b in zip(ScoresLR['L'][1], ScoresLR['R'][0])]
        elif elmpath=='s-L':
            all_scores = [np.asarray(torch.sigmoid(-torch.tensor(a))) for a in ScoresLR['L'][0]]
            all_scores2 = [(a-min(a))/(max(a)-min(a)) for a in all_scores]
        elif elmpath[:4]=='Rs-L':
            mult=float(elmpath[5:])
            all_scores = [np.asarray(torch.sigmoid(-torch.tensor(a))) + mult*b for a ,b in zip(ScoresLR['L'][0], ScoresLR['R'][0])]
            all_scores2 = [(a-min(a))/(max(a)-min(a))+mult*b for a,b in zip(all_scores, ScoresLR['R'][1])]
        elif elmpath[:3]=='RsL':
            mult=float(elmpath[4:])
            all_scores = [np.asarray(torch.sigmoid(torch.tensor(a))) + mult*b for a ,b in zip(ScoresLR['L'][0], ScoresLR['R'][0])]
            all_scores2 = [(a-min(a))/(max(a)-min(a))+b for a,b in zip(all_scores, ScoresLR['R'][1])]
        elif elmpath[:2] == 'sL':
            bias = float(elmpath[3:])
            all_scores = [np.asarray(torch.sigmoid(torch.tensor(a))) * bias for a in ScoresLR['L'][0]]
            all_scores2 = [(a - min(a)) / (max(a) - min(a)) + bias for a in all_scores]
        elif elmpath == 'R-L':
            all_scores = [b - a for a, b in zip(ScoresLR['L'][1], ScoresLR['R'][0])]
            all_scores2 = [b - a for a, b in zip(ScoresLR['L'][1], ScoresLR['R'][1])]
        elif elmpath == 'L-R':
            all_scores = [a - b for a, b in zip(ScoresLR['L'][1], ScoresLR['R'][0])]
            all_scores2 = [a - b for a, b in zip(ScoresLR['L'][1], ScoresLR['R'][1])]
        else:
            print(elmpath)
            all_scores = []
            all_scores2 = []
            all_scores3 = []
            with open(config.name+ '_' + ','.join([str(x) for x in config.mask_ratio_fs]) +",dwsfs_{}".format(config.dw_s_fs)+",IV_{}".format(config.invert_loss)+ '_' + elmpath + '.json') as f:
                data = json.loads(f.read())
                keys = list(data.keys())

                for video_name in keys:
                    scores = np.asarray(data[video_name])
                    scores2 = (scores - min(scores)) / (max(scores) - min(scores))
                    all_scores.append(scores)
                    all_scores2.append(scores2)
                ScoresLR[elmpath] = [all_scores, all_scores2]

        all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
        #with h5py.File(DATASET_PATH, 'r') as hdf:
        for video_name in keys:
            video_index = video_name[6:]

            user_summary = np.array( hdf.get('video_'+video_index+'/user_summary') )
            sb = np.array( hdf.get('video_'+video_index+'/change_points') )
            n_frames = np.array( hdf.get('video_'+video_index+'/n_frames') )
            positions = np.array( hdf.get('video_'+video_index+'/picks') )

            all_user_summary.append(user_summary)
            all_shot_bound.append(sb)
            all_nframes.append(n_frames)
            all_positions.append(positions)

        if config.twoD==1 and elmpath!='R':
            scores_2d=ScoresLR['R'][0]
        elif config.twoD==2 and elmpath!='R':
            scores_2d = ScoresLR['R'][1]
        else:
            scores_2d=[]
        all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions,0, config.nwbP, scores_2d)
        all_summaries2 = generate_summary(all_shot_bound, all_scores2, all_nframes, all_positions,0, config.nwbP, scores_2d)
        if all_scores3:
            all_summaries3 = generate_summary(all_shot_bound, all_scores3, all_nframes, all_positions,0, config.nwbP, scores_2d)

        all_f_scores = []
        all_f_scores2 = []
        all_f_scores3 = []
        # compare the resulting summary with the ground truth one, for each video
        for video_index in range(len(all_summaries)):
            summary = all_summaries[video_index]
            user_summary = all_user_summary[video_index]
            f_score = evaluate_summary(summary, user_summary, 'max' if config.video_type == 'SumMe' else 'avg')
            all_f_scores.append(f_score)
            summary = all_summaries2[video_index]
            f_score = evaluate_summary(summary, user_summary, 'max' if config.video_type == 'SumMe' else 'avg')
            all_f_scores2.append(f_score)
            if all_scores3:
                summary = all_summaries3[video_index]
                f_score = evaluate_summary(summary, user_summary, 'max' if config.video_type == 'SumMe' else 'avg')
                all_f_scores3.append(f_score)
        if all_f_scores3:
            all_f_scores_elm_dict[elmpath] = [all_f_scores, all_f_scores2,all_f_scores3]
            print("f_score_" + elmpath + ": ", np.mean(all_f_scores_elm_dict[elmpath][0]))
            print("f_score_Normilised_R_not_L" + ": ", np.mean(all_f_scores_elm_dict[elmpath][2]))
            print("f_score_Normilised_" + elmpath + ": ", np.mean(all_f_scores_elm_dict[elmpath][1]))

        else:
            all_f_scores_elm_dict[elmpath] = [all_f_scores, all_f_scores2]
            print("f_score_" + elmpath + ": ", np.mean(all_f_scores_elm_dict[elmpath][0]))
            print("f_score_Normilised_" + elmpath + ": ", np.mean(all_f_scores_elm_dict[elmpath][1]))




    log_tp = []
    for v, k in config.__dict__.items():
        if v != 'name' and v != 'name2':
            log_tp.append(k)
    log_tp.append("loss")
    log_tp.append(loss_dic['CB'])

    #for elmpath in ['R','L','sL','R-L','Rs-L','s-L']:
    for elmpath in sclist:
        if elmpath == 'RL':
                log_tp.append('fs_N_L_Not_R')
                log_tp.append(np.mean(all_f_scores_elm_dict[elmpath][2]))
        log_tp.append('fs_N_' + elmpath)
        log_tp.append(np.mean(all_f_scores_elm_dict[elmpath][1]))
        log_tp.append('fs_'+elmpath)
        log_tp.append(np.mean(all_f_scores_elm_dict[elmpath][0]))
        
    reslog(log_tp)