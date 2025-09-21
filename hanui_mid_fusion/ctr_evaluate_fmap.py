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
# from evaluation import compute_eer
import torch
from audiotools import AudioSignal

import nn.loss as losses
from model.discriminator_fmap import Discriminator_fmap
from model.dac import DAC
from model.discriminator import Discriminator

import nn.loss as losses
from collections import defaultdict
# load_checkpoint
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
        #self.generator = Student_DAC().to(self.device)
        self.generator = DAC().to(self.device)
        self.discriminator_pretrained = Discriminator().to(self.device)

        self.rank = rank

        model_parameters = filter(lambda p: p.requires_grad, self.discriminator.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of trainable parameters (M): {params//10000/100}")
        # from ptflops import get_model_complexity_info
        self.chkpt_path = os.path.join(a.checkpoint_path, a.name)
        self.chkpt_path_student = os.path.join(a.checkpoint_path_student, a.name_student)
        self.sampling_rate = 16000
        valid_dataset = MFDataset(a.test_dir, chunk_size=64000, sample_rate=self.sampling_rate)
        self.valid_dataset = DataLoader(
            valid_dataset,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        
        checkpoint = load_checkpoint(self.chkpt_path, self.device)
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        print('Load checkpoint from '+self.chkpt_path)
        
        checkpoint = load_checkpoint(self.chkpt_path_student, self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        print('Load checkpoint from '+self.chkpt_path_student)

        pretrained_disc_path = 'outdir/mix_draft/chkpt/best_model.ckpt-344_d.pt'
        checkpoint = load_checkpoint(pretrained_disc_path, self.device)
        self.discriminator_pretrained.load_state_dict(checkpoint["discriminator"])
        print('Load checkpoint from '+pretrained_disc_path)


        self.sigmoid = nn.Sigmoid()
        weight = torch.FloatTensor([0.1, 0.9]).to(self.device)
        self.ce_loss = nn.CrossEntropyLoss(weight)
        self.save_path = a.savedir
        os.makedirs(self.save_path, exist_ok=True)
        self.index = a.index

    def eval(self):
        self.discriminator.eval()
        self.generator.eval()
        self.discriminator_pretrained.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            print('CtrSVDD Dev EER')
            val_err_01 = self._run_one_epoch(self.valid_dataset)

    def _run_one_epoch(self, data_loader):
        pbar = tqdm(data_loader, total=len(data_loader))
        pred, label = [], []
        target_scores = []
        nontarget_scores = []
        for i, batch in enumerate(pbar):
            sig, key, names = batch
            if '0_T04_139.flac' == names[0] or '1_T04_139.flac' == names[0]: # data error
                continue
            key = key.view(-1).type(torch.int64).to(self.device)
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            sig = AudioSignal(sig, 16000)

            d_real = self.discriminator_pretrained(sig.audio_data)

            d_real_mrd = d_real[5:]
            
            mrd_logit_list = []
            for i in range(len(d_real_mrd)):
                for j in range(len(d_real_mrd[0])-1):
                    mrd_logit_list.append(self.discriminator.discriminators_mrd[i*(len(d_real_mrd[0])-1) + j](d_real_mrd[i][j].detach()))
            
            '''
            loss = []
            for i in range(len(mrd_logit_list)):
                loss.append(self.ce_loss(mrd_logit_list[i], key))
            #breakpoint()
            idx = loss.index(min(loss))
            dict_[idx] += 1
            '''
            #pred_label = mrd_logit_list[self.index]
            pred_label = sum(mrd_logit_list)
            

            #pred_label = F.softmax(pred_label, dim=-1)
            batch_score = (pred_label[:, 1]).data.cpu().numpy().ravel()
            key = key.data.cpu().numpy().ravel()
            # breakpoint()
            for i in range(len(key)):
                if key[i] == 1: # spoof
                    target_scores.append(batch_score[i])
                else:
                    nontarget_scores.append(batch_score[i])
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
        #eer, _ = compute_eer(target_scores, nontarget_scores)
        eer = eer_from_scores(target_scores, nontarget_scores)
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
    parser.add_argument("--test_dir", default=Path('/home/sam/DB/ctrsvdd/test_set/'), type=Path)
    parser.add_argument("--num-workers", default=8, type=int,
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
    

