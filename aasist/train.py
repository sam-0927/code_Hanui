import os
import pdb
import torch
import time
from dataset import MFDataset
# from tflocoformer_small import TFLocoformerSeparator as model
from AASIST import Model as model
import json
# from utils import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from utils import dag_fdd_loss
# from torchcontrib.optim import SWA
from eval_metrics import compute_eer, eer_from_scores
import pdb
import random

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
# load_checkpoint
class Train(object):
    def __init__(self, rank, a):
        torch.cuda.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device('cuda:{:d}'.format(rank))
        # self.model = WavLM().to(self.device)
        self.rank = rank
        self.epochs = a.training_epochs
        with open("AASIST.conf", "r") as f_json:
            config = json.loads(f_json.read())
        model_config = config["model_config"]
       
        # self.model = model(model_config).to(self.device)
        self.model = model().to(self.device)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of trainable parameters (M): {params//10000/100}")
        from ptflops import get_model_complexity_info
        self.chkpt_dir = a.checkpoint_path
        self.optim = torch.optim.Adam(self.model.parameters(), 1e-4)
        # self.optimizer_swa = SWA(self.optim)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=2)
        self.sampling_rate = 16000
        train_dataset = MFDataset(a.train_dir, train=True, chunk_size=64600, \
                                sample_rate=self.sampling_rate) # random_start=True, 
        save_folder = a.checkpoint_path
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=a.num_workers,
            drop_last=True,
        )
        test_dataset = MFDataset(a.test_dir, valid=True, train=False, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        self.tr_loss, self.cv_loss, self.eer_loss = torch.Tensor(self.epochs), torch.Tensor(self.epochs), torch.Tensor(self.epochs)
        for epoch in range(self.epochs):
            self.tr_loss[epoch] = 10000.0
            self.cv_loss[epoch] = 10000.0
            self.eer_loss[epoch] = 10000.0

        self.validation_interval = a.validation_interval
        self.summary_interval = a.summary_interval
        self.writer = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
        self.val_no_impv = 0
        self.log_dir = os.path.join(save_folder, 'log')
        self.chkpt_dir = os.path.join(save_folder, 'chkpt')
        # self.decode_dir = os.path.join(save_folder, 'decode')
        # self.stft_setup = {'filter_length': 128, 'hop_length': 64, 'win_length': 128, 'window_power': 1.}
        
        weight = torch.FloatTensor([0.1, 0.9]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight)
        total_steps = 100 * len(self.train_loader)
        self.scheduler = scheduler = torch.optim.lr_scheduler.LambdaLR(
                                        self.optim,
                                        lr_lambda=lambda step: cosine_annealing(
                                            step,
                                            total_steps,
                                            1,  # since lr_lambda computes multiplicative factor
                                            0.000005 /0.0001))
        self._reset()
        
        

    def save_checkpoint(self, checkpoint_path, epoch, save_trainer=True):
        model_chkpt_path = os.path.join(self.chkpt_dir, checkpoint_path)
        # model_chkpt_path = checkpoint_path
        torch.save({"model": self.model.state_dict()}, model_chkpt_path)
        if logging:
            self.log_msg("Saved checkpoint: {}".format(model_chkpt_path))

        if save_trainer:
            trainer_chkpt_path = model_chkpt_path.replace('model.ckpt', 'trainer.ckpt')
            torch.save({
                        "optimizer": self.optim.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "global_step": self.global_step,
                        "epoch": epoch,
                        "cv_loss": self.cv_loss,
                        "eer_loss": self.eer_loss,
                        }, trainer_chkpt_path)
            if logging:
                self.log_msg("Saved checkpoint: {}".format(trainer_chkpt_path))

        # Save checkpoint list
        #with open(os.path.join(args.checkpoint_dir, 'checkpoint'), 'w') as f:
        with open(os.path.join(self.chkpt_dir, 'model_checkpoint'), 'w') as f:
            f.write(checkpoint_path)
        with open(os.path.join(self.chkpt_dir, 'trainer_checkpoint'), 'w') as f:
            f.write(checkpoint_path.replace('model', 'trainer'))
    
    def _reset(self):
        # Reset
        print(self.device)

        self.writer = SummaryWriter(self.log_dir)
        # Attempt to restore
        last_chkpt = os.path.join(self.chkpt_dir, 'model.ckpt-last.pt')
        print(last_chkpt)
        # chkpt_list = os.path.join(self.chkpt_dir, 'model_checkpoint')
        self.best_val_loss = float("inf")
        self.best_val_eer = float("inf")

        if os.path.exists(last_chkpt) and os.path.exists(last_chkpt.replace('model.ckpt', 'trainer.ckpt')):
            # Loading model (model)
            checkpoint = load_checkpoint(last_chkpt, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.log_msg('Loading checkpoint model %s' % last_chkpt)

            # Loading others (optimizer, discriminator)
            chkpt_path = last_chkpt.replace('model.ckpt', 'trainer.ckpt')
            checkpoint = load_checkpoint(chkpt_path, self.device)
            self.optim.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.log_msg('Loading checkpoint model %s' % chkpt_path)

            self.global_step = checkpoint["global_step"] + 1
            self.start_epoch = checkpoint["epoch"] + 1
            self.cv_loss[:self.start_epoch] = checkpoint["cv_loss"][:self.start_epoch]
            self.best_val_loss = min(self.cv_loss[:self.start_epoch])
            self.eer_loss[:self.start_epoch] = checkpoint["eer_loss"][:self.start_epoch]
            self.best_val_eer = min(self.eer_loss[:self.start_epoch])

        else:
            self.start_epoch = 0
            self.global_step = 0
            os.makedirs(self.chkpt_dir, exist_ok=True)
            with open(f'{self.log_dir}/setups.log', 'w') as fp:
                fp.write('\n%s\n' % (self.model))
                fp.write('\n%s\n' % (self.optim))
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
            self.model.train()
            loss = self._run_one_epoch(train=True)
            self.writer.add_scalar("training/loss_total", loss, self.global_step)

            self.log_msg("Cross validation...")
            self.model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                val_eer = self._run_one_epoch_eval()
                val_loss = self._run_one_epoch(train=False)
            
            with open(f'{self.log_dir}/val_curve.log', 'a') as fp:
                msg = 'Epoch: {}\t validation eer: {:.2f}, loss: {:.5f}\t\n'.format(epoch+1, val_eer, val_loss)
                fp.write(msg)

            self.log_msg(msg)
            self.writer.add_scalar("validation/loss_total", val_loss, self.global_step)
            self.writer.add_scalar("validation/eer", val_eer, self.global_step)

            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = 'ce_best_model.ckpt-{:03d}.pt'.format(epoch + 1)
                trainer_flag = True if epoch > 20 else False
                self.save_checkpoint(file_path, epoch, save_trainer=trainer_flag)
                self.log_msg("Find better validated model, saving to %s" % file_path)

            self.eer_loss[epoch] = val_eer
            if val_eer < self.best_val_eer:
                self.best_val_eer = val_eer
                file_path = 'eer_best_model.ckpt-{:03d}.pt'.format(epoch + 1)
                trainer_flag = True if epoch > 20 else False
                self.save_checkpoint(file_path, epoch, save_trainer=trainer_flag)
                self.log_msg("Find better validated model, saving to %s" % file_path)
           
            file_path = 'model.ckpt-last.pt'.format(epoch + 1)
            self.save_checkpoint(file_path, epoch, save_trainer=True)


    def _run_one_epoch(self, train=True):
        loss_total = []
        if train:
            data_loader = self.train_loader
            self.model.train()
        else:
            data_loader = self.test_loader
            self.model.eval()
        pbar = tqdm(data_loader, total=len(data_loader))
        running_loss = 0
        num_total= 0.0
        #bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # per-sample loss
        #bce_loss = nn.BCEWithLogitsLoss()
        alpha = 0.2
        for i, batch in enumerate(pbar):
            sig, key = batch
            batch_size = sig.size(0)
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            #key = key.view(-1).type(torch.float).to(self.device)
            # breakpoint()
            # pdb.set_trace()
            batch_out = self.model(sig)
            '''##ce training
            if train:
                batch_out = batch_out.requires_grad_(True).squeeze()
            else:
                batch_out = batch_out[0]    # batch_size=1
            '''
            #loss = bce_loss(batch_out, key.float())
            loss = self.criterion(batch_out, key.to(self.device))
            '''
            # Per-sample BCE loss 계산
            losses = bce_loss(batch_out, key.float())
           
            # DAG-FDD CVaR loss 계산
            loss = dag_fdd_loss(losses, alpha)
            '''
            num_total += batch_size
            running_loss += loss.item()
            pbar.set_description("loss: {:.5f}, running loss: {:.5f}".format(
                            loss.item(), running_loss / num_total))
        
            if train:
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optim.step()
                self.scheduler.step()
            
            self.global_step += 1
            loss_total.append(loss.item())
            
        return np.mean(loss_total)
        
    def _run_one_epoch_eval(self):
        data_loader = self.test_loader
        pbar = tqdm(data_loader, total=len(data_loader))
        
        self.model.eval()
        target_scores = []
        nontarget_scores = []

        for i, batch in enumerate(pbar):
            sig, key = batch
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            batch_out = self.model(sig)
            # batch_out = F.softmax(batch_out, dim=-1)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

            key = key.data.cpu().numpy().ravel()

            for i in range(len(key)):
                if key[i] == 1:
                    target_scores.append(batch_score[i])
                else:
                    nontarget_scores.append(batch_score[i])
        #eer = compute_eer(target_scores, nontarget_scores)
        eer = eer_from_scores(target_scores, nontarget_scores)
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

    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default="/ssd2/outputs/singfake/aasist")
    parser.add_argument('--summary_interval', default=1000, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument("--train-dir", default=Path('/ssd2/Database/singfake/mixtures/train/'), type=Path)
    parser.add_argument("--test-dir", default=Path('/ssd2/Database/singfake/mixtures/'), type=Path)
    parser.add_argument("--training-epochs", metavar="NUM_EPOCHS", default=600,
                            type=int, help="The number of epochs to train. (default: 200)")
    parser.add_argument("--num-workers", default=10, type=int,
                            help="The number of workers for dataloader. (default: 4)")
    parser.add_argument("--seed", default=8230, type=int)
    a = parser.parse_args()

    torch.manual_seed(1234)
    solver = Train(0, a)
    solver.train()

if __name__ == '__main__':
    main()
    


