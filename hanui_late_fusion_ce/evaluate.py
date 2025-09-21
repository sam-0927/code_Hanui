import os
import pdb
import torch
import time
from dataset import MFDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from train import load_checkpoint
#from evaluation import compute_eer
import numpy as np
import sklearn.metrics
# from eval_metrics import compute_eer
import torch
from audiotools import AudioSignal

import nn.loss_eval as losses
from model.dac import DAC
from model.discriminator import Discriminator

# load_checkpoint
class Evaluator(object):
    def __init__(self, rank, a):
        torch.cuda.manual_seed(1234)
        self.device = torch.device('cuda:{:d}'.format(rank))
        self.generator = DAC().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.rank = rank

        model_parameters = filter(lambda p: p.requires_grad, self.generator.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of trainable parameters (M): {params//10000/100}")
        # from ptflops import get_model_complexity_info
        self.chkpt_path = os.path.join(a.checkpoint_path, a.name)
        self.disc_path = os.path.join(a.discriminator_path, a.discriminator_name)
        self.sampling_rate = 16000
        # save_folder = a.checkpoint_path
        valid_dataset = MFDataset(a.test_dir,train=False,  valid=True, eval_t01=False, eval_t02=False, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.valid_dataset = DataLoader(
            valid_dataset,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        valid_dataset_spoof = MFDataset(a.test_dir,train=False,  valid=True, eval_t01=False, eval_t02=False, spoof=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.valid_dataset_spoof = DataLoader(
            valid_dataset_spoof,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        test_dataset_01 = MFDataset(a.test_dir,train=False, eval_t01=True, eval_t02=False, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_01 = DataLoader(
            test_dataset_01,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        test_dataset_01_spoof = MFDataset(a.test_dir,train=False, eval_t01=True, eval_t02=False, spoof=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_01_spoof = DataLoader(
            test_dataset_01_spoof,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        test_dataset_02 = MFDataset(a.test_dir,train=False, eval_t01=False, eval_t02=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_02 = DataLoader(
            test_dataset_02,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        test_dataset_02_spoof = MFDataset(a.test_dir,train=False, eval_t01=False, eval_t02=True, spoof=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_02_spoof = DataLoader(
            test_dataset_02_spoof,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        
        test_dataset_04 = MFDataset(a.test_dir,train=False, eval_t04=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_04 = DataLoader(
            test_dataset_04,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        test_dataset_04_spoof = MFDataset(a.test_dir,train=False, eval_t04=True, spoof=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_04_spoof = DataLoader(
            test_dataset_04_spoof,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        
        test_dataset_03 = MFDataset(a.test_dir,train=False, eval_t03=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_03 = DataLoader(
            test_dataset_03,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        test_dataset_03_spoof = MFDataset(a.test_dir,train=False, eval_t03=True, spoof=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_03_spoof = DataLoader(
            test_dataset_03_spoof,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        
        
        self.waveform_loss = losses.L1Loss()
        self.stft_loss = losses.MultiScaleSTFTLoss()
        self.mel_loss = losses.MelSpectrogramLoss()
        self.gan_loss = losses.GANLoss(self.discriminator)

        checkpoint = load_checkpoint(self.chkpt_path, self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        print('Load checkpoint from '+self.chkpt_path)
        
        checkpoint = load_checkpoint(self.disc_path, self.device)
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        print('Load checkpoint from '+self.disc_path)
        # self.sigmoid = nn.Sigmoid()
        # weight = torch.FloatTensor([0.1, 0.9]).to(self.device)
        self.save_path = a.savedir
        os.makedirs(self.save_path, exist_ok=True)

    def eval(self):
        self.generator.eval()
        torch.cuda.empty_cache()
        val_loss_total = []
        with torch.no_grad():
            '''
            # stft, mel, wav = self._run_one_epoch(self.valid_dataset)
            fmap = self._run_one_epoch(self.valid_dataset)
            print('validataion Bonafide')
            # # print('stft loss: {}, mel loss: {}, waveform loss: {}\n'.format(stft, mel, wav))
            print('feature map btw GT and Recon: {}\n'.format(fmap))
            
            # # stft, mel, wav = self._run_one_epoch(self.valid_dataset_spoof)
            fmap = self._run_one_epoch(self.valid_dataset_spoof)
            print('validataion Spoof')
            # # print('stft loss: {}, mel loss: {}, waveform loss: {}\n'.format(stft, mel, wav))
            print('feature map btw GT and Recon: {}\n'.format(fmap))
            
            # # stft, mel, wav = self._run_one_epoch(self.test_loader_01)
            fmap = self._run_one_epoch(self.test_loader_01)
            print('T01 Bonafide')
            # # print('stft loss: {}, mel loss: {}, waveform loss: {}\n'.format(stft, mel, wav))
            print('feature map btw GT and Recon: {}\n'.format(fmap))
            '''
            # stft, mel, wav = self._run_one_epoch(self.test_loader_01_spoof)
            fmap = self._run_one_epoch(self.test_loader_01_spoof)
            print('T01 Spoof')
            # print('stft loss: {}, mel loss: {}, waveform loss: {}\n'.format(stft, mel, wav))
            print('feature map btw GT and Recon: {}\n'.format(fmap))
            
            # stft, mel, wav = self._run_one_epoch(self.test_loader_02)
            fmap = self._run_one_epoch(self.test_loader_02)
            print('T02 Bonafide')
            # print('stft loss: {}, mel loss: {}, waveform loss: {}\n'.format(stft, mel, wav))
            print('feature map btw GT and Recon: {}\n'.format(fmap))
            
            # stft, mel, wav = self._run_one_epoch(self.test_loader_02_spoof)
            fmap = self._run_one_epoch(self.test_loader_02_spoof)
            print('T02 Spoof')
            # print('stft loss: {}, mel loss: {}, waveform loss: {}\n'.format(stft, mel, wav))
            print('feature map btw GT and Recon: {}\n'.format(fmap))
            
            # stft, mel, wav = self._run_one_epoch(self.test_loader_04)
            fmap = self._run_one_epoch(self.test_loader_04)
            print('T04 Bonafide')
            # print('stft loss: {}, mel loss: {}, waveform loss: {}\n'.format(stft, mel, wav))
            print('feature map btw GT and Recon: {}\n'.format(fmap))
            
            # stft, mel, wav = self._run_one_epoch(self.test_loader_04_spoof)
            fmap = self._run_one_epoch(self.test_loader_04_spoof)
            print('T04 Spoof')
            # print('stft loss: {}, mel loss: {}, waveform loss: {}\n'.format(stft, mel, wav))
            print('feature map btw GT and Recon: {}\n'.format(fmap))
            
            # stft, mel, wav = self._run_one_epoch(self.test_loader_03)
            fmap = self._run_one_epoch(self.test_loader_03)
            print('T03 Bonafide')
            # print('stft loss: {}, mel loss: {}, waveform loss: {}\n'.format(stft, mel, wav))
            print('feature map btw GT and Recon: {}\n'.format(fmap))
            
            # stft, mel, wav = self._run_one_epoch(self.test_loader_03_spoof)
            fmap = self._run_one_epoch(self.test_loader_03_spoof)
            print('T03 Spoof')
            # print('stft loss: {}, mel loss: {}, waveform loss: {}'.format(stft, mel, wav))
            print('feature map btw GT and Recon: {}\n'.format(fmap))
            
            # print('Validation EER')
            # print('T01 EER')
            # val_err_01 = self._run_one_epoch(self.test_loader_01)
            # print('T02 EER')
            # val_err_02 = self._run_one_epoch(self.test_loader_02)
       

    def _run_one_epoch(self, data_loader):
        pbar = tqdm(data_loader, total=len(data_loader))
        pred, label = [], []
        target_scores = []
        nontarget_scores = []
        stft_loss_total, mel_loss_total, waveform_loss_total = [], [], []
        fmap_loss_total = []
        for i, batch in enumerate(pbar):
            sig, key, names = batch
            if '0_T04_139.flac' == names[0] or '1_T04_139.flac' == names[0]: # data error
                continue
            
            batch_size = sig.size(0)
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            sig = sig.unsqueeze(1)
            #key = key.view(-1).type(torch.float).to(self.device)
            
            recons = self.generator(sig, self.sampling_rate)
            
            recons = AudioSignal(recons, 16000)
            sig = AudioSignal(sig, 16000)
            
            # Discriminator feature map loss
            _, feat_loss, mpd_loss, mrd_loss = self.gan_loss.generator_loss(recons, sig)
            fmap_loss_total.append(feat_loss.item() / batch_size)
            stft_loss_total.append(mpd_loss.item() / batch_size)
            mel_loss_total.append(mrd_loss.item() / batch_size)
        print('MPD: {}'.format(np.mean(stft_loss_total)))
        print('MRD: {}'.format(np.mean(mel_loss_total)))
        return np.mean(fmap_loss_total)
            # # Save waveform
            # for i in range(len(names)):
            #     recons.detach().cpu().write(self.save_path+'/recon_'+names[i].split('.')[0]+'.wav')
            #     sig.detach().cpu().write(self.save_path+'/gt_'+names[i].split('.')[0]+'.wav')
            
            # stft_loss = self.stft_loss(recons, sig)
            # mel_loss = self.mel_loss(recons, sig)
            # waveform_loss = self.waveform_loss(recons, sig)
            # stft_loss_total.append(stft_loss.item() / batch_size)
            # mel_loss_total.append(mel_loss.item() / batch_size)
            # waveform_loss_total.append(waveform_loss.item() / batch_size)
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
    parser.add_argument('--checkpoint_path', default="outdir/mix_all/chkpt")
    parser.add_argument('--name', default='model.ckpt-last_g.pt')
    parser.add_argument('--summary_interval', default=1000, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument("--test_dir", default=Path('/home/sam/DB/singfake/mixtures/'), type=Path)
    parser.add_argument("--num-workers", default=10, type=int,
                            help="The number of workers for dataloader. (default: 4)")
    parser.add_argument("--seed", default=8230, type=int)
    parser.add_argument("--savedir", default='savedir', type=str)
    parser.add_argument('--discriminator_path', default='outdir/mix_all/chkpt', type=str)
    parser.add_argument('--discriminator_name', default='model.ckpt-last_d.pt')
    a = parser.parse_args()

    torch.manual_seed(1234)
    solver = Evaluator(0, a)
    solver.eval()

if __name__ == '__main__':
    main()
    

