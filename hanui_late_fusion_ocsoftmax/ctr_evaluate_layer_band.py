import os
import pdb
import torch
import time
from dataset_ctr import MFDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sklearn.metrics
from eval_metrics import compute_eer, eer_from_scores

import torch
from audiotools import AudioSignal

import nn.loss as losses
from model.discriminator_fmap import Discriminator_fmap
from model.dac import DAC
from model.discriminator import Discriminator

import nn.loss as losses
from collections import defaultdict
import loss

def load_checkpoint(filepath, device):
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

class Evaluator(object):
    def __init__(self, rank, a):
        torch.cuda.manual_seed(1234)
        self.device = torch.device('cuda:{:d}'.format(rank))
        self.discriminator = Discriminator_fmap().to(self.device)
        self.generator = DAC().to(self.device)
        self.discriminator_pretrained = Discriminator().to(self.device)

        self.rank = rank

        model_parameters = filter(lambda p: p.requires_grad, self.discriminator.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of trainable parameters (M): {params//10000/100}")
        # from ptflops import get_model_complexity_info
        self.chkpt_path = os.path.join(a.checkpoint_path, a.name)
        self.chkpt_path_student = os.path.join(a.checkpoint_path_student, a.name_student)
        self.chkpt_path_disc = os.path.join(a.checkpoint_path_student_disc, a.name_student_disc)
        self.sampling_rate = 16000
        valid_dataset = MFDataset(a.test_dir, chunk_size=64000, sample_rate=self.sampling_rate)
        self.valid_dataset = DataLoader(
            valid_dataset,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        dev_dataset = MFDataset(a.dev_dir, chunk_size=64000, sample_rate=self.sampling_rate)
        self.dev_dataset = DataLoader(
            dev_dataset,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )

        
        checkpoint = load_checkpoint(self.chkpt_path, self.device)
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        print('Load checkpoint from '+self.chkpt_path)
        self.ocsoftmax_loss = loss.OCSoftmax(feat_dim=32).cuda()
        self.ocsoftmax_loss.load_state_dict(checkpoint["ocsoftmax"])
        
        checkpoint = load_checkpoint(self.chkpt_path_student, self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        print('Load checkpoint from '+self.chkpt_path_student)

        #pretrained_disc_path = 'outdir/mix_draft/chkpt/best_model.ckpt-344_d.pt'
        checkpoint = load_checkpoint(self.chkpt_path_disc, self.device)
        self.discriminator_pretrained.load_state_dict(checkpoint["discriminator"])
        print('Load checkpoint from '+self.chkpt_path_disc)


        self.sigmoid = nn.Sigmoid()
        weight = torch.FloatTensor([0.1, 0.9]).to(self.device)
        self.ce_loss = nn.CrossEntropyLoss(weight)
        self.save_path = a.savedir
        os.makedirs(self.save_path, exist_ok=True)
        self.index = a.index
        

    def get_mrd_band_layer(self, idx):
        mrd = idx // 25 + 1
        band = (idx % 25) // 5 + 1
        layer = (idx % 5) + 1
        return mrd, band, layer

    def eval(self):
        self.discriminator.eval()
        self.ocsoftmax_loss.eval()
        self.generator.eval()
        self.discriminator_pretrained.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            print('Test EER')
            val_err_01 = self._run_one_epoch(self.valid_dataset)
            print('Dev EER')
            val_err_01 = self._run_one_epoch(self.dev_dataset)

    def _run_one_epoch(self, data_loader):
        pbar = tqdm(data_loader, total=len(data_loader))
        pred, label = [], []
        target_scores = []
        nontarget_scores = []
        scores_loader, idx_loader = [], []
        for i, batch in enumerate(pbar):
            sig, key, names = batch
            if '0_T04_139.flac' == names[0] or '1_T04_139.flac' == names[0]: # data error
                continue
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            sig = sig.unsqueeze(1)
            with torch.no_grad():
                recons = self.generator(sig, self.sampling_rate)
                recons = recons.detach()
                
                recons = AudioSignal(recons, 16000)
                sig = AudioSignal(sig, 16000)
                d_fake = self.discriminator_pretrained(recons.audio_data)
                d_real = self.discriminator_pretrained(sig.audio_data)
            d_real_mrd = d_real[5:]
            d_fake_mrd = d_fake[5:]
            
            mrd_logit_list = []
            for i in range(len(d_fake_mrd)):
                for j in range(len(d_fake_mrd[0])-1):
                    mrd_logit_list.append(self.discriminator.discriminators_mrd[i*(len(d_fake_mrd[0])-1) + j](d_real_mrd[i][j].detach(), d_fake_mrd[i][j].detach()))
            summed_logit_mrd = sum(mrd_logit_list)  # [B, 32]
            disc_mrd_loss, score = self.ocsoftmax_loss(summed_logit_mrd, key)
            scores_loader.append(score)
            idx_loader.append((key))

        scores = torch.cat(scores_loader, 0).data.cpu().numpy()
        labels = torch.cat(idx_loader, 0).data.cpu().numpy()
        eer = eer_from_scores(scores[labels==0], scores[labels==1])
        print(eer)
        return eer
            


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
    parser.add_argument('--checkpoint_path_student', default="outdir/mix_draft/chkpt")
    parser.add_argument('--name_student', default='best_model.ckpt-344_g.pt')
    parser.add_argument('--checkpoint_path_student_disc', default="outdir/mix_draft/chkpt")
    parser.add_argument('--name_student_disc', default='best_model.ckpt-344_d.pt')

    parser.add_argument('--summary_interval', default=1000, type=int)
    parser.add_argument("--test_dir", default=Path('/ssd2/Database/ctrsvdd/test_set_4sec/'), type=Path)
    parser.add_argument("--dev_dir", default=Path('/ssd2/Database/ctrsvdd/dev_set_4sec/'), type=Path)

    parser.add_argument("--num-workers", default=0, type=int,
                            help="The number of workers for dataloader. (default: 4)")
    parser.add_argument("--seed", default=8230, type=int)
    parser.add_argument("--savedir", default='savedir', type=str)
    parser.add_argument("--index", default=0, type=int)
    a = parser.parse_args()

    torch.manual_seed(1234)
    solver = Evaluator(0, a)
    solver.eval()

if __name__ == '__main__':
    main()
    

