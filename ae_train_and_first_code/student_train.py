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
from audiotools import AudioSignal

import nn.loss as losses
from model.dac import DAC
from model.dac_student import Student_DAC
from model.discriminator import Discriminator
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
        self.teacher = DAC().to(self.device)
        self.generator = Student_DAC().to(self.device)
        # self.discriminator = Discriminator().to(self.device)
        self.rank = rank
        self.epochs = a.training_epochs

        model_parameters = filter(lambda p: p.requires_grad, self.generator.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of trainable parameters (M): {params//10000/100}")
        self.optim_g = torch.optim.AdamW(self.generator.parameters(), 1e-4)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, 1.0)
        # self.optim_d = torch.optim.AdamW(self.discriminator.parameters(), 1e-4)
        # self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, 1.0)
        
        self.sampling_rate = 16000
        train_dataset = MFDataset(a.train_dir, train=True, chunk_size=6400, \
                                sample_rate=self.sampling_rate) # random_start=True, 
        model_folder = os.path.join(a.output_dir, a.checkpoint_path)
        teacher_model_folder = os.path.join(a.teacher_dir, a.teacher_checkpoint_path)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=a.num_workers,
            drop_last=True,
        )
        test_dataset = MFDataset(a.test_dir,train=False, valid=True, chunk_size=64000, \
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
        self.teacher_dir = os.path.join(teacher_model_folder, 'chkpt')
        
        self._reset()
        # weight = torch.FloatTensor([0.1, 0.9]).to(self.device)
        # self.criterion = nn.CrossEntropyLoss(weight)
        self.waveform_loss = losses.L1Loss()
        self.stft_loss = losses.MultiScaleSTFTLoss()
        self.mel_loss = losses.MelSpectrogramLoss()
        # self.gan_loss = losses.GANLoss(self.discriminator)
        self.ts_loss = losses.TSLoss()
        

    def save_checkpoint(self, checkpoint_path, epoch, save_trainer=True):
        model_chkpt_path = os.path.join(self.chkpt_dir, checkpoint_path)
        g_chkpt_path = model_chkpt_path.split('.pt')[0] + '_g.pt'
        # d_chkpt_path = model_chkpt_path.split('.pt')[0] + '_d.pt'
        torch.save({
                        "generator":self.generator.state_dict(),
                        "optimizer": self.optim_g.state_dict(),
                        "scheduler": self.scheduler_g.state_dict(),
                        "global_step": self.global_step,
                        "epoch": epoch,
                        "cv_loss": self.cv_loss,
                        }, g_chkpt_path)
        # torch.save({
        #                 "discriminator":self.discriminator.state_dict(),
        #                 "optimizer": self.optim_d.state_dict(),
        #                 "scheduler": self.scheduler_d.state_dict(),
        #                 }, d_chkpt_path)
        if logging:
            self.log_msg("Saved checkpoint: {}".format(model_chkpt_path))

        # Save checkpoint list
        with open(os.path.join(self.chkpt_dir, 'model_checkpoint'), 'w') as f:
            f.write(checkpoint_path)

    
    def _reset(self):
        # Reset
        print(self.device)
        # Load Teacher model
        teacher_chkpt_g = os.path.join(self.teacher_dir, 'best_model.ckpt-344_g.pt')
        if os.path.exists(teacher_chkpt_g):
            teacher_checkpoint = load_checkpoint(teacher_chkpt_g, self.device)
            self.teacher.load_state_dict(teacher_checkpoint['generator'])
            self.log_msg('Loading Teacher checkpoint model %s' % teacher_chkpt_g)
        self.writer = SummaryWriter(self.log_dir)
        # Attempt to restore
        last_chkpt_g = os.path.join(self.chkpt_dir, 'model.ckpt-last_g.pt')
        # last_chkpt_d = os.path.join(self.chkpt_dir, 'model.ckpt-last_d.pt')
        self.prev_val_loss, self.best_val_loss = float("inf"), float("inf")
        
        if os.path.exists(last_chkpt_g):
            checkpoint = load_checkpoint(last_chkpt_g, self.device)
            self.generator.load_state_dict(checkpoint["generator"])
            self.optim_g.load_state_dict(checkpoint["optimizer"])
            self.scheduler_g.load_state_dict(checkpoint["scheduler"])
            self.global_step = checkpoint["global_step"] + 1
            self.start_epoch = checkpoint["epoch"] + 1
            self.cv_loss[:self.start_epoch] = checkpoint["cv_loss"][:self.start_epoch]
            self.best_val_loss = min(self.cv_loss[:self.start_epoch])
            self.log_msg('Loading checkpoint model %s' % last_chkpt_g)
        # if os.path.exists(last_chkpt_d):
        #     checkpoint = load_checkpoint(last_chkpt_d, self.device)
        #     self.discriminator.load_state_dict(checkpoint["discriminator"])
        #     self.optim_d.load_state_dict(checkpoint["optimizer"])
        #     self.scheduler_d.load_state_dict(checkpoint["scheduler"])
        #     self.log_msg('Loading checkpoint model %s' % last_chkpt_d)
        else:
            self.start_epoch = 0
            self.global_step = 0
            os.makedirs(self.chkpt_dir, exist_ok=True)
            with open(f'{self.log_dir}/setups.log', 'w') as fp:
                fp.write('\n%s\n' % (self.generator))
                # fp.write('\n%s\n' % (self.discriminator))
                fp.write('\n%s\n' % (self.optim_g))
                # fp.write('\n%s\n' % (self.optim_d))
                fp.write('\n%s\n' % (self.scheduler_g))
                # fp.write('\n%s\n' % (self.scheduler_d))
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
            self.generator.train()
            # self.discriminator.train()
            self.teacher.eval()
            _, _, _ = self._run_one_epoch(train=True)            

            self.log_msg("Cross validation...")
            self.generator.eval()
            # self.discriminator.eval()
            torch.cuda.empty_cache()
            val_loss_total = []
            with torch.no_grad():
                mel_loss, ref, recon = self._run_one_epoch(train=False)
                val_loss_total.append(mel_loss)
            
            self.writer.add_audio('valid/gt', ref[0].audio_data, self.global_step, 16000)
            self.writer.add_audio('valid/recon', recon[0].audio_data, self.global_step, 16000)
            self.writer.add_figure('valid/gt_mel', plot_melspectrogram(ref[0]), self.global_step)
            self.writer.add_figure('valid/recon_mel', plot_melspectrogram(recon[0]), self.global_step)
            
            val_loss_total = np.mean(val_loss_total)
            # if epoch+1 > 85:
            #     self.scheduler.step(val_loss_total)
            with open(f'{self.log_dir}/val_curve.log', 'a') as fp:
                msg = 'Epoch: {}\t validation loss: {:.5f}\t\n'.format(epoch+1, val_loss_total)
                fp.write(msg)

            self.log_msg(msg)
            # self.writer.add_scalar("validation/loss_total", val_loss_total, self.global_step)
            self.cv_loss[epoch] = val_loss_total
            if self.global_step % 50000 == 0:
                file_path = 'model.ckpt-{:03d}.pt'.format(epoch + 1)
                self.save_checkpoint(file_path, epoch, save_trainer=True)
            if val_loss_total < self.best_val_loss:
                self.best_val_loss = val_loss_total
                file_path = 'best_model.ckpt-{:03d}.pt'.format(epoch + 1)
                trainer_flag = True if epoch > 20 else False
                self.save_checkpoint(file_path, epoch, save_trainer=trainer_flag)
                self.log_msg("Find better validated model, saving to %s" % file_path)
            
            file_path = 'model.ckpt-last.pt'.format(epoch + 1)
            self.save_checkpoint(file_path, epoch, save_trainer=True)


    def _run_one_epoch(self, train=True):
        loss_total = []
        if train:
            data_loader = self.train_loader
        else:
            data_loader = self.test_loader
        pbar = tqdm(data_loader, total=len(data_loader))
        total_gen_loss = 0
        total_disc_loss = 0
        num_total= 0.0
        self.avg_loss = 0

        # disc_list = []
        # stft_list, mel_list, waveform_list, gen_list, feat_list = [],[],[],[],[]
        
        # loss_logs = {
        #     "disc": [],
        #     "stft": [],
        #     "mel": [],
        #     "waveform": [],
        #     "gen": [],
        #     "feat": [],
        # }
        # loss_names = ["disc", "stft", "mel", "waveform", "gen", "feat"]
        loss_logs = {
            "waveform": [],
            "stft": [],
            "enc": [],
            "dec": [],
        }
        loss_names = ["waveform", "stft", "enc", "dec"]

        for i, batch in enumerate(pbar):
            sig, key, names = batch
            batch_size = sig.size(0)
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            sig = sig.unsqueeze(1)
            
            with torch.no_grad():
                recons_t, encoder_list_t, decoder_list_t = self.teacher.forward_save_output(sig, self.sampling_rate)
            recons_s, encoder_list_s, decoder_list_s = self.generator(sig, self.sampling_rate)
            
            if(sig.size() != recons_s.size() or recons_t.size() != recons_s.size()):
                breakpoint()
                
            recons_s = AudioSignal(recons_s, 16000)
            sig = AudioSignal(sig, 16000)
            
            # enc: [4,32,64,6400] = [n_layer, B, dim(layer마다 다름), T(layer마다 다름)]
            # dec: [4,32,1536,25] = [n_layer, B, dim(layer마다 다름), T(layer마다 다름)]
            ts_loss_enc = self.ts_loss(encoder_list_s, encoder_list_t)
            ts_loss_dec = self.ts_loss(decoder_list_s, decoder_list_t)
            # disc_loss = self.gan_loss.discriminator_loss(recons, sig)
            # if train:
            #     self.optim_d.zero_grad()
            #     disc_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
            #     self.optim_d.step()
            #     self.scheduler_d.step()

            stft_loss = self.stft_loss(recons_s, sig)
            # mel_loss = self.mel_loss(recons, sig)
            # waveform_loss = self.waveform_loss(recons_s, sig)
            waveform_loss = self.waveform_loss(recons_s,  AudioSignal(recons_t, 16000).detach())
            # gen_loss, feat_loss = self.gan_loss.generator_loss(recons, sig)
            
            # generator_loss = 1*waveform_loss + 1*ts_loss_enc + 2*ts_loss_dec
            generator_loss = 10*ts_loss_enc + ts_loss_dec
            
            # generator_loss = stft_loss + 15*mel_loss + waveform_loss + 1*gen_loss + 2*feat_loss
                
            if train:
                self.optim_g.zero_grad()
                generator_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1e3)
                self.optim_g.step()
                self.scheduler_g.step()
  
            num_total += batch_size
            
            # 각 loss를 딕셔너리에 기록
            loss_values = {
                # "disc": disc_loss,
                # "mel": mel_loss,
                "waveform": waveform_loss,
                "stft": stft_loss,
                # "gen": gen_loss,
                # "feat": feat_loss,
                "enc": ts_loss_enc,
                "dec": ts_loss_dec,
            }
            for k in loss_logs:
                loss_logs[k].append(loss_values[k].item())
            # disc_list.append(disc_loss.item())
            # stft_list.append(stft_loss.item())
            # mel_list.append(mel_loss.item())
            # waveform_list.append(waveform_loss.item())
            # gen_list.append(gen_loss.item())
            # feat_list.append(feat_loss.item())
            
            # total_gen_loss += generator_loss.item()
            # total_disc_loss += disc_loss.item()
            
            pbar.set_description("Enc loss: {:.5f}, Dec loss: {:.5f}, waveform loss: {:.5f} stft loss: {:.5f}".format(ts_loss_enc.item(), ts_loss_dec.item(), waveform_loss.item(), stft_loss.item()))
            
            self.global_step += 1
            
        # avg_losses = tuple(np.mean(loss_logs[k]) / batch_size for k in ["disc", "stft", "mel", "waveform", "gen", "feat"])
        avg_losses = tuple(np.mean(loss_logs[k]) / batch_size for k in ["waveform", "stft", "enc", "dec"])
        if train:
            self.writer.add_audio('train/gt', sig[0].audio_data, self.global_step, 16000)
            self.writer.add_audio('train/recon', recons_s[0].audio_data, self.global_step, 16000)
            self.writer.add_figure('train/gt_mel', plot_melspectrogram(sig[0]), self.global_step)
            self.writer.add_figure('train/recon_mel', plot_melspectrogram(recons_s[0]), self.global_step)
            for name, avg in zip(loss_names, avg_losses):
                self.writer.add_scalar(f"training/{name}_loss", avg, self.global_step)
        else:
            for name, avg in zip(loss_names, avg_losses):
                self.writer.add_scalar(f"validation/{name}_loss", avg, self.global_step)
        
        return avg_losses[0], sig, recons_s

        # return np.mean(disc_list)/batch_size, np.mean(stft_list)/batch_size, np.mean(mel_list)/batch_size, np.mean(waveform_list)/batch_size, np.mean(gen_list)/batch_size, np.mean(feat_list)/batch_size, \
                # sig, recons
        # return total_gen_loss/num_total , total_disc_loss/num_total, sig, recons
        

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
    parser.add_argument('--teacher_dir', default="outdir")
    parser.add_argument('--checkpoint_path', default="student_draft")
    parser.add_argument('--teacher_checkpoint_path', default="mix_draft")
    parser.add_argument('--summary_interval', default=1000, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument("--train_dir", default=Path('/home/sam/SingFake/dataset/split_dump_re/mixtures/'), type=Path)
    parser.add_argument("--test_dir", default=Path('/home/sam/SingFake/dataset/split_dump_re/mixtures/'), type=Path)
    parser.add_argument("--training-epochs", metavar="NUM_EPOCHS", default=600,
                            type=int, help="The number of epochs to train. (default: 200)")
    parser.add_argument("--num-workers", default=10, type=int,
                            help="The number of workers for dataloader. (default: 4)")
    a = parser.parse_args()

    torch.manual_seed(1234)
    solver = Train(0, a)
    solver.train()

if __name__ == '__main__':
    main()
    

