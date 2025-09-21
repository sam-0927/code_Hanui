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
from audiotools import AudioSignal

import nn.loss as losses
from model.dac import DAC
# from model.dac_student import Student_DAC
# from model.discriminator_student_2 import Discriminator_student_2
from model.discriminator_latent import Discriminator_latent
from utils import plot_melspectrogram

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# load_checkpoint
class Train(object):
    def __init__(self, rank, a):
        torch.cuda.manual_seed(1234)
        self.device = torch.device('cuda:{:d}'.format(rank))
        self.discriminator = Discriminator_latent().to(self.device)
        # self.generator = Student_DAC().to(self.device)
        self.generator = DAC().to(self.device)
        self.rank = rank
        self.epochs = a.training_epochs

        model_parameters = filter(lambda p: p.requires_grad, self.discriminator.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of trainable parameters (M): {params//10000/100}")
        self.optim_d = torch.optim.AdamW(self.discriminator.parameters(), 1e-4)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, 1.0)
        
        self.sampling_rate = 16000
        train_dataset = MFDataset(a.train_dir, train=True, disc=True, chunk_size=6400, \
                                sample_rate=self.sampling_rate) # random_start=True, 
        model_folder = os.path.join(a.output_dir, a.checkpoint_path)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=a.num_workers,
            drop_last=True,
        )
        test_dataset = MFDataset(a.test_dir, train=False, valid=True, disc=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        self.tr_loss, self.cv_loss = torch.Tensor(self.epochs), torch.Tensor(self.epochs)
        for epoch in range(self.epochs):
            self.tr_loss[epoch] = 10000.0
            self.cv_loss[epoch] = 10000.0

        self.validation_interval = a.validation_interval
        self.summary_interval = a.summary_interval

        self.val_no_impv = 0
        self.log_dir = os.path.join(model_folder, 'log')
        self.chkpt_dir = os.path.join(model_folder, 'chkpt')
        
        self._reset()
        weight = torch.FloatTensor([0.1, 0.9]).to(self.device)
        self.ce_loss = nn.CrossEntropyLoss(weight)
        self.start_valid_spoof = False
        self.start_valid_bona = False
        
        self.chkpt_path_student = os.path.join(a.checkpoint_path_student, a.name_student)
        checkpoint = load_checkpoint(self.chkpt_path_student, self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        print('Load checkpoint from '+self.chkpt_path_student)
        self.generator.eval()
        # self.waveform_loss = losses.L1Loss()
        # self.stft_loss = losses.MultiScaleSTFTLoss()
        # self.mel_loss = losses.MelSpectrogramLoss()
        # self.gan_loss = losses.GANLoss(self.discriminator)
        # self.ts_loss = losses.TSLoss()
        

    def save_checkpoint(self, checkpoint_path, epoch, save_trainer=True):
        model_chkpt_path = os.path.join(self.chkpt_dir, checkpoint_path)
        d_chkpt_path = model_chkpt_path.split('.pt')[0] + '_d.pt'
        torch.save({
                        "discriminator":self.discriminator.state_dict(),
                        "optimizer": self.optim_d.state_dict(),
                        "scheduler": self.scheduler_d.state_dict(),
                        "global_step": self.global_step,
                        "epoch": epoch,
                        "cv_loss": self.cv_loss,
                        }, d_chkpt_path)
        if logging:
            self.log_msg("Saved checkpoint: {}".format(model_chkpt_path))

        # Save checkpoint list
        with open(os.path.join(self.chkpt_dir, 'model_checkpoint'), 'w') as f:
            f.write(checkpoint_path)

    
    def _reset(self):
        # Reset
        print(self.device)
        self.writer = SummaryWriter(self.log_dir)
        # Attempt to restore
        last_chkpt_d = os.path.join(self.chkpt_dir, 'model.ckpt-last_d.pt')
        self.prev_val_loss, self.best_val_loss = float("inf"), float("inf")
        
        if os.path.exists(last_chkpt_d):
            checkpoint = load_checkpoint(last_chkpt_d, self.device)
            self.discriminator.load_state_dict(checkpoint["discriminator"])
            self.optim_d.load_state_dict(checkpoint["optimizer"])
            self.scheduler_d.load_state_dict(checkpoint["scheduler"])
            self.global_step = checkpoint["global_step"] + 1
            self.start_epoch = checkpoint["epoch"] + 1
            self.cv_loss[:self.start_epoch] = checkpoint["cv_loss"][:self.start_epoch]
            self.best_val_loss = min(self.cv_loss[:self.start_epoch])
            self.log_msg('Loading checkpoint model %s' % last_chkpt_d)
        else:
            self.start_epoch = 0
            self.global_step = 0
            os.makedirs(self.chkpt_dir, exist_ok=True)
            with open(f'{self.log_dir}/setups.log', 'w') as fp:
                fp.write('\n%s\n' % (self.discriminator))
                fp.write('\n%s\n' % (self.optim_d))
                fp.write('\n%s\n' % (self.scheduler_d))
            with open(f'{self.log_dir}/val_curve.log', 'w') as fp:
                fp.write('')
    @staticmethod
    def log_msg(msg):
        print(msg)
        logging.info(msg)

    def update_average(self, loss, avg_loss, step):
        """Update running average of the loss.

        G-FDD lossDAG-FDD lossDAG-FDD loss
        ---------
        loss : torch.tensor
            detached loss, a single float value.
        avg_loss : float
            current running average.

        Returns
        -------
        avg_loss : float
            The average loss.
        """
        if torch.isfinite(loss):
            avg_loss -= avg_loss / step
            avg_loss += float(loss) / step
        return avg_loss

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.log_msg(f'Epoch: {epoch + 1}')
            self.log_msg("Training...")
            self.discriminator.train()
            _ = self._run_one_epoch(train=True)
            
            self.log_msg("Cross validation...")
            self.discriminator.eval()
            torch.cuda.empty_cache()
            val_loss_total = []
            best_mrd_loss_total = []
            with torch.no_grad():
                ce_loss, best_mrd_loss = self._run_one_epoch(train=False)
                val_loss_total.append(ce_loss)
                best_mrd_loss_total.append(best_mrd_loss)
            
            val_loss_total = np.mean(val_loss_total)
            best_mrd_loss_total = np.mean(best_mrd_loss_total)
            with open(f'{self.log_dir}/val_curve.log', 'a') as fp:
                msg = 'Epoch: {}\t validation loss: {:.5f}\t\n'.format(epoch+1, val_loss_total)
                fp.write(msg)

            self.log_msg(msg)
            # self.writer.add_scalar("validation/loss_total", val_loss_total, self.global_step)
            self.cv_loss[epoch] = val_loss_total
            if self.global_step % 50000 == 0:
                file_path = 'model.ckpt-{:03d}.pt'.format(epoch + 1)
                self.save_checkpoint(file_path, epoch, save_trainer=True)
            if best_mrd_loss_total < self.best_val_loss:
                self.best_val_loss = best_mrd_loss_total
                file_path = 'best_model.ckpt-{:03d}.pt'.format(epoch + 1)
                trainer_flag = True if epoch > 20 else False
                self.save_checkpoint(file_path, epoch, save_trainer=trainer_flag)
                self.log_msg("Find better validated model, saving to %s" % file_path)
            
            file_path = 'model.ckpt-last.pt'.format(epoch + 1)
            self.save_checkpoint(file_path, epoch, save_trainer=True)


    def _run_one_epoch(self, train=True):
        if train:
            data_loader = self.train_loader
        else:
            data_loader = self.test_loader
        pbar = tqdm(data_loader, total=len(data_loader))
        
        num_total= 0.0
        self.avg_loss = 0
        
        loss_logs = {
            "disc": [],
            # "mpd_1": [],
            # "mpd_2": [],
            # "mpd_3": [],
            # "msd": [],
            "mrd_combine": [],
        }
        loss_names = ['disc', 'mrd_combine']
        # loss_names = ['disc', 'mrd_freq_1', 'mrd_freq_2', 'mrd_freq_3', 'mrd_freq_4', 'mrd_freq_5']
        for i, batch in enumerate(pbar):
            sig, key, names = batch
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            
            sig = sig.unsqueeze(1)
            with torch.no_grad():
                # recons,_,_ = self.generator(sig, self.sampling_rate)
                recons = self.generator.encode(sig, self.sampling_rate)
                # recons = self.generator(recons, self.sampling_rate)
            
            recons = recons.detach()
            # recons = AudioSignal(recons, 16000)
            # sig = AudioSignal(sig, 16000)
            
            key = key.view(-1).type(torch.int64).to(self.device)
            # if sig.audio_data.size() != recons.audio_data.size():
            #     breakpoint()

            # pred_label_log = self.discriminator.forward_logit_log(sig.audio_data, recons.audio_data) # MRD 5 layer결과. stft를 log-scale로 함. [B,5,2]
            # pred_label += pred_label_log
            # pred_label = self.discriminator.forward_logit(sig.audio_data, recons.audio_data)
            pred_label = self.discriminator.forward_logit(recons)
            
            if not train:
                pred_label = pred_label.unsqueeze(0)
            disc_loss = self.ce_loss(pred_label, key)
            disc_losses = []
            disc_losses.append(disc_loss)
            
            # disc_losses = []
            # disc_loss = 0
            # for i in range(len(pred_label)):
            #     # if not train:
            #     #     pred_label[i] = pred_label[i].unsqueeze(0)
            #     loss = self.ce_loss(pred_label[i].squeeze(-1), key)
            #     disc_loss += loss
            #     disc_losses.append(loss)

            if train:
                self.optim_d.zero_grad()
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
                self.optim_d.step()
                self.scheduler_d.step()
                
            # 각 loss를 딕셔너리에 기록
            loss_values = {
                "disc": disc_loss,
                # "mpd_1": disc_losses[0],
                # "mpd_2": disc_losses[1],
                # "mpd_3": disc_losses[2],
                # "msd": disc_losses[3],
                "mrd_combine": disc_losses[0],
            }
            for k in loss_logs:
                loss_logs[k].append(loss_values[k].item())
            
            pbar.set_description("CE loss: {:.5f}".format(disc_loss.item()/len(pred_label)))
            
            self.global_step += 1
            # if not train: 
            #     if not self.start_valid_spoof:
            #         if key[0] == 1:
            #             self.writer.add_audio('valid/ref_spoof', sig[0].audio_data, self.global_step, 16000)
            #             self.writer.add_figure('valid/ref_spoof_mel', plot_melspectrogram(sig[0]), self.global_step)
            #             self.writer.add_audio('valid/recon_spoof', recons[0].audio_data, self.global_step, 16000)
            #             self.writer.add_figure('valid/recon_spoof_mel', plot_melspectrogram(recons[0]), self.global_step)
            #             self.start_valid_spoof = True
            #     if not self.start_valid_bona:
            #         if key[0] == 0:
            #             self.writer.add_audio('valid/ref_bonafide', sig[0].audio_data, self.global_step, 16000)
            #             self.writer.add_figure('valid/ref_bonafide_mel', plot_melspectrogram(sig[0]), self.global_step)
            #             self.writer.add_audio('valid/recon_bonafide', recons[0].audio_data, self.global_step, 16000)
            #             self.writer.add_figure('valid/recon_bonafide_mel', plot_melspectrogram(recons[0]), self.global_step)
            #             self.start_valid_bona = True
            
        # avg_losses = tuple(np.mean(loss_logs[k]) / batch_size for k in ['mpd_1','mpd_2','mpd_3','mpd_4','mpd_5', 'msd','mrd_1','mrd_2','mrd_3'])
        avg_losses = tuple(np.mean(loss_logs[k]) for k in ["disc", 'mrd_combine'])
        # avg_losses = tuple(np.mean(loss_logs[k]) for k in ["disc",'mrd_freq_1','mrd_freq_2','mrd_freq_3','mrd_freq_4','mrd_freq_5'])
        
        if train:
            for name, avg in zip(loss_names, avg_losses):
                self.writer.add_scalar(f"training/{name}_loss", avg, self.global_step)                
        else:
            for name, avg in zip(loss_names, avg_losses):
                self.writer.add_scalar(f"validation/{name}_loss", avg, self.global_step)
        
        return avg_losses[0]/len(pred_label), avg_losses[1]/len(pred_label) 


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

    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default="outdir")
    parser.add_argument('--checkpoint_path', default="discriminator")
    parser.add_argument('--checkpoint_path_student', default="/home/sam/SingFake/models/dac/outdir/mix_draft/chkpt")
    parser.add_argument('--name_student', default='best_model.ckpt-344_g.pt')
    # parser.add_argument('--checkpoint_path_student', default="/home/sam/SingFake/models/dac/outdir/mix_student/chkpt")
    # parser.add_argument('--name_student', default='model.ckpt-last_g.pt')

    parser.add_argument('--summary_interval', default=1000, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument("--train_dir", default=Path('/home/sam/SingFake/dataset/split_dump_re/mixtures/'), type=Path)
    # parser.add_argument("--recon_dir", default=Path('/home/sam/SingFake/dataset/split_dump_re/student_make/'), type=Path)
    parser.add_argument("--test_dir", default=Path('/home/sam/SingFake/dataset/split_dump_re/mixtures/'), type=Path)
    parser.add_argument("--training-epochs", metavar="NUM_EPOCHS", default=600,
                            type=int, help="The number of epochs to train. (default: 200)")
    parser.add_argument("--num-workers", default=0, type=int,
                            help="The number of workers for dataloader. (default: 4)")
    a = parser.parse_args()

    torch.manual_seed(1234)
    solver = Train(0, a)
    solver.train()

if __name__ == '__main__':
    main()
    

