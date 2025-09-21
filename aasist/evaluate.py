import os
import pdb
import torch
import time
from dataset import MFDataset
from AASIST import Model as model
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
from eval_metrics import compute_eer, eer_from_scores
# from eval_metrics_sy import compute_eer
# from evaluate_tDCF_asvspoof19_resnet import compute_eer_and_tdcf
import torch
from loss import *
import pdb
# load_checkpoint
class Evaluator(object):
    def __init__(self, rank, a):
        torch.cuda.manual_seed(1234)
        self.device = torch.device('cuda:{:d}'.format(rank))
        self.model = model().to(self.device)
        self.rank = rank

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of trainable parameters (M): {params//10000/100}")
        from ptflops import get_model_complexity_info
        self.chkpt_path = os.path.join(a.checkpoint_path, a.name)
        self.sampling_rate = 16000
        save_folder = a.checkpoint_path
        valid_dataset = MFDataset(a.test_dir,train=False,valid=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.valid_dataset = DataLoader(
            valid_dataset,
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
        test_dataset_02 = MFDataset(a.test_dir,train=False, eval_t01=False, eval_t02=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_02 = DataLoader(
            test_dataset_02,
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
        test_dataset_04 = MFDataset(a.test_dir,train=False, eval_t04=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_04 = DataLoader(
            test_dataset_04,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        test_dataset_ctrtest = MFDataset(a.test_dir,train=False, ctr_test=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_ctrtest = DataLoader(
            test_dataset_ctrtest,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )
        test_dataset_ctrdev = MFDataset(a.test_dir,train=False, ctr_dev=True, chunk_size=64000, \
                                sample_rate=self.sampling_rate)
        self.test_loader_ctrdev = DataLoader(
            test_dataset_ctrdev,
            batch_size=1,
            num_workers=a.num_workers,
            drop_last=False,
        )


        # self.stft_setup = {'filter_length': 128, 'hop_length': 64, 'win_length': 128, 'window_power': 1.}
        checkpoint = load_checkpoint(self.chkpt_path, self.device)
        self.model.load_state_dict(checkpoint["model"])
        print('Load checkpoint from '+self.chkpt_path)
        self.sigmoid = nn.Sigmoid()

        loss_model_path = self.chkpt_path.replace("best_model.ckpt-136.pt", "anti-spoofing_loss_model.pt")
        loss = 'ocsoftmax'
        self.loss_model = torch.load(loss_model_path)
        # weight = torch.FloatTensor([0.1, 0.9]).to(self.device)

    def eval(self):
        self.model.eval()
        torch.cuda.empty_cache()
        val_loss_total = []
        with torch.no_grad():
            '''
            print('T04 EER')
            val_err_04 = self._run_one_epoch(self.test_loader_04)
            print('Validation EER')
            valid_eer = self._run_one_epoch(self.valid_dataset)
            print('T01 EER')
            val_err_01 = self._run_one_epoch(self.test_loader_01)
            print('T02 EER')
            val_err_02 = self._run_one_epoch(self.test_loader_02)
            print('T03 EER')
            val_err_03 = self._run_one_epoch(self.test_loader_03)
            '''
            print('Ctr Test EER')
            val_err_03 = self._run_one_epoch(self.test_loader_ctrtest)
            print('Ctr Dev EER')
            val_err_03 = self._run_one_epoch(self.test_loader_ctrdev)
     

    def _run_one_epoch(self, data_loader):
        pbar = tqdm(data_loader, total=len(data_loader))
        target_scores = []
        nontarget_scores = []
        for i, batch in enumerate(pbar):
            sig, key = batch
            batch_size = sig.size(0)
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            #key = key.view(-1).type(torch.float).to(self.device)
            batch_out = self.model(sig)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

            key = key.data.cpu().numpy().ravel()
            # ang_isoloss, score = self.loss_model(batch_score, key)
            for i in range(len(key)):
                if key[i] == 1:
                    target_scores.append(batch_score[i])
                else:
                    nontarget_scores.append(batch_score[i])
        #eer = compute_eer(target_scores, nontarget_scores)
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
    parser.add_argument('--checkpoint_path', default="/home/sam/SingFake/models/output/tfloco_small/chkpt")
    parser.add_argument('--name', default='best_model.ckpt-136.pt')
    parser.add_argument('--summary_interval', default=1000, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument("--test-dir", default=Path('/ssd2/Database/singfake/mixtures/'), type=Path)
    parser.add_argument("--training-epochs", metavar="NUM_EPOCHS", default=200,
                            type=int, help="The number of epochs to train. (default: 200)")
    parser.add_argument("--num-workers", default=10, type=int,
                            help="The number of workers for dataloader. (default: 4)")
    parser.add_argument("--seed", default=8230, type=int)
    a = parser.parse_args()

    torch.manual_seed(1234)
    solver = Evaluator(0, a)
    solver.eval()

if __name__ == '__main__':
    main()
    


