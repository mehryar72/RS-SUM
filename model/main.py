from configs import get_config
from solver import Solver

import logging
import os
import csv
import torch
from data_loader import VideoData, Records
# from torch.utils.tensorboard import SummaryWriter

def reslog(line):
    with open(r'/home/mabbasib/Transformer_pretr/results/acc.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(line)
        f.close()


if __name__ == '__main__':
    config = get_config()

    logF = config.name + '.csv'
    # tb_writer = SummaryWriter(config.name2)
    print(config)
    if (os.path.isfile(logF)):
        os.remove(logF)
    logging.basicConfig(filename=logF,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger('myloger')

    # logger.info(
    #     ',tr_loss,te_loss,minloss,lr')
    #print('yo11')
    if config.video_type == 'CB' and config.full_Len == 0:
        trainset1 = VideoData('train', config.split_index,
                              name='summe',
                              dw_s=config.dw_s, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                              Masking=config.Masking,
                              Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio,
                              mask_chance=config.mask_chance, replace_chance=config.replace_chance,
                              overlap=config.overlap,
                              window_s=config.window_s,
                              nwB=config.nwB)
        trainset2 = VideoData('train', config.split_index,
                              name='tvsum',
                              dw_s=config.dw_s, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                              Masking=config.Masking,
                              Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio,
                              mask_chance=config.mask_chance, replace_chance=config.replace_chance,
                              overlap=config.overlap,
                              window_s=config.window_s,
                              nwB=config.nwB)
        trainset = torch.utils.data.ConcatDataset([trainset1, trainset2])
        testset1 = VideoData('test', config.split_index,
                             name='summe',
                             dw_s=config.dw_s_te, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                             Masking=config.Masking,
                             Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio_te,
                             mask_chance=config.mask_chance, replace_chance=config.replace_chance,
                             overlap=config.overlap,
                             window_s=config.window_s_te,
                             DlTyp=config.DlTyp_te,
                              nwB=config.nwB)
        testset2 = VideoData('test', config.split_index,
                             name='tvsum',
                             dw_s=config.dw_s_te, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                             Masking=config.Masking,
                             Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio_te,
                             mask_chance=config.mask_chance, replace_chance=config.replace_chance,
                             overlap=config.overlap,
                             window_s=config.window_s_te,
                             DlTyp=config.DlTyp_te,
                              nwB=config.nwB)
        testset = torch.utils.data.ConcatDataset([testset1, testset2])
    else:
        trainset = VideoData('train', config.split_index,
                             name='summe' if config.video_type == 'SumMe' else 'tvsum',
                             dw_s=config.dw_s, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                             Masking=config.Masking,
                             Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio,
                             mask_chance=config.mask_chance, replace_chance=config.replace_chance,
                             overlap=config.overlap,
                             window_s=config.window_s,
                              nwB=config.nwB)
        if config.CombD == 1 and config.full_Len == 0:
            trainset1 = VideoData('train', config.split_index,
                                  name='tvsum' if config.video_type == 'SumMe' else 'summe',
                                  dw_s=config.dw_s, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                                  Masking=config.Masking,
                                  Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio,
                                  mask_chance=config.mask_chance, replace_chance=config.replace_chance,
                                  overlap=config.overlap,
                                  window_s=config.window_s,
                                  nwB=config.nwB)
            trainset2 = VideoData('train', config.split_index,
                                  name='tvsum' if config.video_type == 'SumMe' else 'summe',
                                  dw_s=config.dw_s, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                                  Masking=config.Masking,
                                  Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio,
                                  mask_chance=config.mask_chance, replace_chance=config.replace_chance,
                                  overlap=config.overlap,
                                  window_s=config.window_s,
                                  CombD=1,
                                  nwB=config.nwB)
            trainset = torch.utils.data.ConcatDataset([trainset, trainset1, trainset2])
        if (config.CombD == 2 or config.CombD == 3) and config.full_Len == 0:
            trainset1 = VideoData('train', config.split_index,
                                  name='tvsum' if config.video_type == 'SumMe' else 'summe',
                                  dw_s=config.dw_s, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                                  Masking=config.Masking,
                                  Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio,
                                  mask_chance=config.mask_chance, replace_chance=config.replace_chance,
                                  overlap=config.overlap,
                                  window_s=config.window_s,
                                  nwB=config.nwB)
            trainset2 = VideoData('train', config.split_index,
                                  name='tvsum' if config.video_type == 'SumMe' else 'summe',
                                  dw_s=config.dw_s, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                                  Masking=config.Masking,
                                  Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio,
                                  mask_chance=config.mask_chance, replace_chance=config.replace_chance,
                                  overlap=config.overlap,
                                  window_s=config.window_s,
                                  CombD=1,
                                  nwB=config.nwB
                                  )
            trainset3 = VideoData('train', 0,
                                  name='ovp',
                                  dw_s=config.dw_s, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                                  Masking=config.Masking,
                                  Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio,
                                  mask_chance=config.mask_chance, replace_chance=config.replace_chance,
                                  overlap=config.overlap,
                                  window_s=config.window_s,
                                  nwB=1)
            trainset4 = VideoData('train', 0,
                                  name='youtube',
                                  dw_s=config.dw_s, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                                  Masking=config.Masking,
                                  Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio,
                                  mask_chance=config.mask_chance, replace_chance=config.replace_chance,
                                  overlap=config.overlap,
                                  window_s=config.window_s,
                                  nwB=1
                                  )
            if config.CombD == 2:
                trainset = torch.utils.data.ConcatDataset([trainset, trainset1, trainset2, trainset3, trainset4])
            elif config.CombD == 3:
                trainset = torch.utils.data.ConcatDataset([trainset1, trainset2, trainset3, trainset4])
        testset = VideoData('test', config.split_index,
                            name='summe' if config.video_type == 'SumMe' else 'tvsum',
                            dw_s=config.dw_s_te, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                            Masking=config.Masking,
                            Full_shot_mask=config.Full_shot_mask, mask_ratio=config.mask_ratio_te,
                            mask_chance=config.mask_chance, replace_chance=config.replace_chance, overlap=config.overlap,
                            window_s=config.window_s_te,
                            DlTyp=config.DlTyp_te)
    testset2 = VideoData('test', config.split_index,
                        name='summe' if config.video_type == 'SumMe' else 'tvsum',
                        dw_s=3, full_Len=config.full_Len, Tr_seq_Len=config.Tr_seq_Len,
                        Masking=config.Masking,
                        Full_shot_mask=config.Full_shot_mask, mask_ratio=[0.15],
                        mask_chance=config.mask_chance, replace_chance=config.replace_chance, overlap=config.overlap,
                        window_s=[1],
                        DlTyp=config.DlTyp_te)
    trainLoader = torch.utils.data.DataLoader(trainset,
                                              batch_size=config.batch_size, shuffle=True, num_workers=0)
    testLoader = torch.utils.data.DataLoader(testset,
                                             batch_size=config.batch_size, shuffle=False, num_workers=0)
    dataset_lens = {'train': len(trainset), 'test': len(testset)}
    solver = Solver(config, trainLoader, testLoader, Records['Max_Len'], logger, dataset_lens,testset2)
    solver.build()
    out_records = solver.train()

    log_tp = []
    for v, k in config.__dict__.items():
        if v != 'name' and v != 'name2':
            log_tp.append(k)
    log_tp.append(out_records[0])
    for record in out_records[1:]:
        for v, k in record.items():
            log_tp.append(k)

    reslog(log_tp)
