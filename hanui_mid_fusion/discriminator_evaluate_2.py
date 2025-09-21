import os
import pdb
import torch
import time
from dataset_2 import MFDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import sklearn.metrics
from eval_metrics import compute_eer
# from evaluation import compute_eer
import torch
from audiotools import AudioSignal

import nn.loss as losses
from model.discriminator_student_2 import Discriminator_student_2
#from model.dac_student import Student_DAC
from model.dac import DAC

import nn.loss as losses

# load_checkpoint
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

class Evaluator(object):
    def __init__(self, rank, a):
        torch.cuda.manual_seed(1234)
        self.device = torch.device('cuda:{:d}'.format(rank))
        self.discriminator = Discriminator_student_2().to(self.device)
        #self.generator = Student_DAC().to(self.device)
        self.generator = DAC().to(self.device)

        self.rank = rank

        model_parameters = filter(lambda p: p.requires_grad, self.discriminator.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of trainable parameters (M): {params//10000/100}")
        # from ptflops import get_model_complexity_info
        self.chkpt_path = os.path.join(a.checkpoint_path, a.name)
        self.chkpt_path_student = os.path.join(a.checkpoint_path_student, a.name_student)
        self.sampling_rate = 16000
        valid_dataset = MFDataset(a.test_dir,train=False,  valid=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.valid_dataset = DataLoader(
            valid_dataset,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        test_dataset_01 = MFDataset(a.test_dir,train=False, eval_t01=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        test_dataset_01_spoof = MFDataset(a.test_dir,train=False, eval_t01=True, spoof=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_01 = DataLoader(
            test_dataset_01 + test_dataset_01_spoof,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        test_dataset_02 = MFDataset(a.test_dir,train=False, eval_t02=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        test_dataset_02_spoof = MFDataset(a.test_dir,train=False, eval_t02=True, spoof=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_02 = DataLoader(
            test_dataset_02 + test_dataset_02_spoof,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        
        test_dataset_04 = MFDataset(a.test_dir,train=False, eval_t04=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        test_dataset_04_spoof = MFDataset(a.test_dir,train=False, eval_t04=True, spoof=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_04 = DataLoader(
            test_dataset_04 + test_dataset_04_spoof,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        self.disc_num = a.disc_num
        test_dataset_03 = MFDataset(a.test_dir,train=False, eval_t03=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        test_dataset_03_spoof = MFDataset(a.test_dir,train=False, eval_t03=True, spoof=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_03 = DataLoader(
            test_dataset_03 + test_dataset_03_spoof,
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
        self.sigmoid = nn.Sigmoid()
        weight = torch.FloatTensor([0.1, 0.9]).to(self.device)
        self.ce_loss = nn.CrossEntropyLoss(weight)
        self.save_path = a.savedir
        os.makedirs(self.save_path, exist_ok=True)

    def eval(self):
        self.discriminator.eval()
        self.generator.eval()
        torch.cuda.empty_cache()
        # val_loss_total = []
        with torch.no_grad():
            print('********** '+str(self.disc_num)+' **********')
            print('Validation EER')
            val_err_01 = self._run_one_epoch(self.valid_dataset)
            print('T01 EER')
            val_err_01 = self._run_one_epoch(self.test_loader_01)
            print('T02 EER')
            val_err_02 = self._run_one_epoch(self.test_loader_02)
            print('T04 EER')
            val_err_04 = self._run_one_epoch(self.test_loader_04)
            print('T03 EER')
            val_err_04 = self._run_one_epoch(self.test_loader_03)
       

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
            
            sig = sig.unsqueeze(1)
            #recons,_,_ = self.generator(sig, self.sampling_rate)
            recons = self.generator(sig, self.sampling_rate)
            recons = self.generator(recons, self.sampling_rate)


            recons = recons.detach()
            recons = AudioSignal(recons, 16000)
            
            sig = AudioSignal(sig, 16000)
            
            # ['mpd_1','mpd_2','mpd_3', 'msd','mrd_1','mrd_2','mrd_3','mrd_4','mrd_5','mrd_1_log','mrd_2_log','mrd_3_log','mrd_4_log','mrd_5_log']
            # ##################### list 중에 결과 보고 뽑던가 해야함. 
            pred_label = self.discriminator.forward_logit_combine(sig.audio_data, recons.audio_data)
            # pred_label_log = self.discriminator.forward_logit_log(sig.audio_data, recons.audio_data) # MRD 5 layer결과. stft를 log-scale로 함. [B,5,2]
            # pred_label += pred_label_log
            # pred_label = pred_label_log
            # pred_label = pred_label[self.disc_num]
            
            pred_label = F.softmax(pred_label, dim=-1)
            batch_score = (pred_label.unsqueeze(0)[:, 1]).data.cpu().numpy().ravel()
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
        eer, _ = compute_eer(target_scores, nontarget_scores)
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
    parser.add_argument('--checkpoint_path_student', default="/home/sam/SingFake/models/dac/outdir/mix_student/chkpt")
    parser.add_argument('--name_student', default='model.ckpt-last_g.pt')
    parser.add_argument('--summary_interval', default=1000, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument("--train_dir", default=Path('/home/sam/SingFake/dataset/split_dump_re/mixtures/'), type=Path)
    parser.add_argument("--test_dir", default=Path('/home/sam/SingFake/dataset/split_dump_re/mixtures/'), type=Path)
    parser.add_argument("--training-epochs", metavar="NUM_EPOCHS", default=200,
                            type=int, help="The number of epochs to train. (default: 200)")
    parser.add_argument("--num-workers", default=8, type=int,
                            help="The number of workers for dataloader. (default: 4)")
    parser.add_argument("--seed", default=8230, type=int)
    parser.add_argument("--savedir", default='savedir', type=str)
    parser.add_argument("--disc_num", default=4, type=int)
    a = parser.parse_args()

    torch.manual_seed(1234)
    solver = Evaluator(0, a)
    solver.eval()

if __name__ == '__main__':
    main()
    

