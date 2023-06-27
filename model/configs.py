# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pprint
import math
import sys
import os

save_dir = Path('../exp1')


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, name,name2, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.name = name
        self.name2 = name2
        # self.termination_point = math.floor(0.15 * self.action_state_size)
        # self.set_dataset_dir(self.video_type)

    # def set_dataset_dir(self, video_type='TVSum'):
    #     self.log_dir = save_dir.joinpath(video_type, 'logs/split' + str(self.split_index))
    #     self.score_dir = save_dir.joinpath(video_type, 'results/split' + str(self.split_index))
    #     self.save_dir = save_dir.joinpath(video_type, 'models/split' + str(self.split_index))


def __repr__(self):
    """Pretty-print configurations in alphabetical order"""
    config_str = 'Configurations\n'
    config_str += pprint.pformat(self.__dict__)
    return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initialized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    # parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--verbose', type=str2bool, default='true')
    parser.add_argument('--video_type', type=str, default='TVSum')
    parser.add_argument('--JID', type=int, default=1)

    # Model
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--F_atn', type=str2bool, default='true')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--posE', type=int, default=0)  # 1 absolute, #2 reletive

    # dataloader Masking
    parser.add_argument('--split_index', type=int, default=0)

    parser.add_argument('--dw_s', type=int, default=8)
    parser.add_argument('--dw_s_te', type=int, default=3)
    parser.add_argument('--overlap', type=int, default=1)
    parser.add_argument('--full_Len', type=int, default=0)
    parser.add_argument('--CombD', type=int, default=0)
    parser.add_argument('--Masking', type=int, default=1)
    parser.add_argument('--Tr_seq_Len', type=int, default=128)
    parser.add_argument('--window_s', type=int, default=8)
    parser.add_argument('--window_s_te', type=int, nargs="+", default=[2])
    parser.add_argument('--window_s_fs', type=int, nargs="+", default=[1])
    parser.add_argument('--Full_shot_mask', type=int, default=1)

    parser.add_argument('--mask_ratio', type=float, default=0.85)
    parser.add_argument('--mask_ratio_te', type=float, nargs="+", default=[0.15,0.40,0.60,0.85])
    parser.add_argument('--mask_ratio_fs', type=float, nargs="+", default=[0.15])
    parser.add_argument('--mask_chance', type=float, default=0.8)
    parser.add_argument('--replace_chance', type=float, default=0.1)

     #fscore
    parser.add_argument('--twoD', type=int, default=0)
    parser.add_argument('--nwB', type=int, default=0)
    parser.add_argument('--nwbP', type=float, default=0.8)
    
    # Train
    parser.add_argument('--n_epochs', type=int, default=250)
    #parser.add_argument('--n_epochs', type=int, default=250)
    parser.add_argument('--losstype', type=str, default='CB')
    parser.add_argument('--Sc', type=int, default=1)
    #parser.add_argument('--Sc', type=int, default=1)
    parser.add_argument('--opt', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    #parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--clip', type=float, default=5.0)
    ##addon
    parser.add_argument('--DlTyp_te', type=int, default=2)
    parser.add_argument('--DlTyp_fs', type=int, default=0)
    parser.add_argument('--invert_loss', type=int, default=0)
    parser.add_argument('--dw_s_fs', type=int, default=3)
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]
    if kwargs.full_Len:
        name2 = '/scratch/mabbasib/Transformer_pretr/tb_logs_hp2/' + kwargs.video_type + '/split_{}/'.format(kwargs.split_index)+'le_1_'+'Los_'+kwargs.losstype+'/'
    else:
        name2 = '/scratch/mabbasib/Transformer_pretr/tb_logs_hp2/' + kwargs.video_type + '/split_{}/'.format(kwargs.split_index)+'le_{}_'.format(kwargs.Tr_seq_Len)+'Los_'+kwargs.losstype+'/'
    if not os.path.exists(name2):
        os.makedirs(name2)
    name = '/scratch/mabbasib/TR_LOGS_test/' + kwargs.video_type + '/split_{}/'.format(kwargs.split_index)
    if not os.path.exists(name):
        os.makedirs(name)
    
    argv_list=[]
    argv_list_flag=1
    for sysargv in sys.argv[1:]:
        if sysargv[0]=='-':
            if sysargv in ['--window_s_fs', '--mask_ratio_fs','--twoD','--nwbP','--DlTyp_fs','--invert_loss','--dw_s_fs']:
                argv_list_flag=0
            else:
                argv_list_flag = 1
        if argv_list_flag==1:
            argv_list.append(sysargv[2:] if sysargv[0]=='-' else sysargv)
    name = name + 'logs' + ','.join(argv_list)
    # name2 =name2 + 'tb_out' + ''.join(sys.argv[1:])
    name2 = name2 + 'tb_out' + ','.join(argv_list)
    #name = name + 'logs' + ''.join(sys.argv[1:])
    #name2 =name2 + 'tb_out' + ''.join(sys.argv[1:])
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(name=name,name2=name2, **kwargs)


if __name__ == '__main__':
    config = get_config()
    import ipdb

    ipdb.set_trace()
