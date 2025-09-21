import os
import pdb
import torch
import time
from dataset import MFDataset
# from architecture import WavLM as model
# from utils import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from train import load_checkpoint
import numpy as np
import sklearn.metrics
# from eval_metrics import compute_eer
import torch
from audiotools import AudioSignal

import nn.loss as losses
from model.dac_student import Student_DAC
from model.dac import DAC
from model.discriminator import Discriminator
import nn.loss as losses

# load_checkpoint
class Evaluator(object):
    def __init__(self, rank, a):
        torch.cuda.manual_seed(1234)
        self.device = torch.device('cuda:{:d}'.format(rank))
        self.generator = Student_DAC().to(self.device)
        self.rank = rank

        model_parameters = filter(lambda p: p.requires_grad, self.generator.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of trainable parameters (M): {params//10000/100}")
        # from ptflops import get_model_complexity_info
        self.chkpt_path = os.path.join(a.checkpoint_path, a.name)
        self.sampling_rate = 16000
        dataset = MFDataset(a.test_dir, train=False,  valid=True, eval_t01=False, eval_t02=False, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.dataset = DataLoader(
            dataset,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        
        
        self.waveform_loss = losses.L1Loss()
        self.stft_loss = losses.MultiScaleSTFTLoss(log_weight=1.0, mag_weight=2.0)
        self.mel_loss = losses.MelSpectrogramLoss(log_weight=1.0, mag_weight=2.0)

        checkpoint = load_checkpoint(self.chkpt_path, self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        print('Load checkpoint from '+self.chkpt_path)
        self.sigmoid = nn.Sigmoid()
        self.save_path = a.savedir
        os.makedirs(self.save_path, exist_ok=True)

    def eval(self):
        self.generator.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            self._run_one_epoch(self.dataset)


    def _run_one_epoch(self, data_loader):
        pbar = tqdm(data_loader, total=len(data_loader))
        # stft_loss_total, mel_loss_total, waveform_loss_total = [], [], []
        per_dict = {}
        with open('../../dataset/metadata/metadata_T04_only_per.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                per_name = line.strip().split('|')[0]
                per_dict[per_name] = per_name
                
        for i, batch in enumerate(pbar):
            sig, key, names = batch
            if '0_T04_139.flac' == names[0] or '1_T04_139.flac' == names[0]: # data error
                continue
            
            # batch_size = sig.size(0)
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            sig = sig.unsqueeze(1)
            #key = key.view(-1).type(torch.float).to(self.device)
            
            recons,_,_ = self.generator(sig, self.sampling_rate)
            
            recons = AudioSignal(recons, 16000)
            # sig = AudioSignal(sig, 16000)
            
            # Save waveform
            for i in range(len(names)):
                if 'T04' in names[i]:
                    if names[i] in per_dict:
                        recons.detach().cpu().write(self.save_path+'/recon_'+names[i].split('.')[0]+'.wav')
                else:
                    recons.detach().cpu().write(self.save_path+'/recon_'+names[i].split('.')[0]+'.wav')
                # sig.detach().cpu().write(self.save_path+'/gt_'+names[i].split('.')[0]+'.wav')
        #     if recons.audio_data.size() != sig.audio_data.size():
        #         breakpoint()
        #     stft_loss = self.stft_loss(recons, sig)
        #     mel_loss = self.mel_loss(recons, sig)
        #     waveform_loss = self.waveform_loss(recons, sig)
        #     stft_loss_total.append(stft_loss.item() / batch_size)
        #     mel_loss_total.append(mel_loss.item() / batch_size)
        #     waveform_loss_total.append(waveform_loss.item() / batch_size)
        # return np.mean(stft_loss_total), np.mean(mel_loss_total), np.mean(waveform_loss_total)
            
            
        # batch_out = self.model(sig)
        # batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # key = key.data.cpu().numpy().ravel()
        # for i in range(len(key)):
        #     if key[i] == 1:
        #         target_scores.append(batch_score[i])
        #     else:
        #         nontarget_scores.append(batch_score[i])
        '''
        score = self.sigmoid(batch_out)
        if key.item() > 0:
            target_scores.append(score.item())
        else:
            nontarget_scores.append(score.item())
        '''
        # pred.append(self.sigmoid(batch_out).item())
        
        # label.append(key.item())
    # eer = compute_eer(label, pred)
    # eer = compute_eer(target_scores, nontarget_scores)
    # print(eer)
    # return eer
        


def main():
    import json
    import argparse
    from pathlib import Path
    import shutil

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self


    def build_env(config, config_name, path):
        t_path = os.path.join(path, config_name)
        if config != t_path:
            os.makedirs(path, exist_ok=True)
            shutil.copyfile(config, os.path.join(path, config_name))

    print('Initializing Evaluation Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default="/home/sam/SingFake/models/dac/model_name/chkpt")
    parser.add_argument('--name', default='model.ckpt-last_g.pt')
    parser.add_argument('--summary_interval', default=1000, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument("--train-dir", default=Path('/home/sam/SingFake/dataset/split_dump_re/vocals/'), type=Path)
    parser.add_argument("--test-dir", default=Path('/home/sam/SingFake/dataset/split_dump_re/vocals/'), type=Path)
    parser.add_argument("--training-epochs", metavar="NUM_EPOCHS", default=200,
                            type=int, help="The number of epochs to train. (default: 200)")
    parser.add_argument("--num-workers", default=10, type=int,
                            help="The number of workers for dataloader. (default: 4)")
    parser.add_argument("--seed", default=8230, type=int)
    parser.add_argument("--savedir", default='savedir', type=str)
    a = parser.parse_args()

    torch.manual_seed(1234)
    solver = Evaluator(0, a)
    solver.eval()

if __name__ == '__main__':
    main()
    

