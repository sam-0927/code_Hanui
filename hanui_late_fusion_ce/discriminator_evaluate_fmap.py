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
from eval_metrics import compute_eer, eer_from_scores
# from evaluation import compute_eer
import torch
from audiotools import AudioSignal

import nn.loss as losses
from model.discriminator import Discriminator
from model.discriminator_fmap import Discriminator_fmap
from model.dac import DAC

import nn.loss as losses
from analysis.mmd import draw_mmd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    check_dict = torch.load(filepath, map_location=device)
    print("complete.")
    return check_dict

class Evaluator(object):
    def __init__(self, rank, a):
        torch.cuda.manual_seed(1234)
        self.device = torch.device('cuda:{:d}'.format(rank))
        self.discriminator = Discriminator_fmap().to(self.device)

        self.discriminator_pretrained = Discriminator().to(self.device)
        self.generator = DAC().to(self.device)
        self.rank = rank

        model_parameters = filter(lambda p: p.requires_grad, self.discriminator.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of trainable parameters (M): {params//10000/100}")
        # from ptflops import get_model_complexity_info
        self.chkpt_path = os.path.join(a.checkpoint_path, a.name)
        self.chkpt_path_student = os.path.join(a.checkpoint_path_student, a.name_student)
        self.chkpt_path_student_disc = os.path.join(a.checkpoint_path_student, a.name_disc)

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
            batch_size=20,
            num_workers=a.num_workers,
            drop_last=False,
            shuffle=True,
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
        self.test_loader_03 = DataLoader(
            test_dataset_03,
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

        checkpoint = load_checkpoint(self.chkpt_path_student_disc, self.device)
        self.discriminator_pretrained.load_state_dict(checkpoint["discriminator"])
        print('Load checkpoint from '+self.chkpt_path_student_disc)
        
        self.sigmoid = nn.Sigmoid()
        weight = torch.FloatTensor([0.1, 0.9]).to(self.device)
        self.ce_loss = nn.CrossEntropyLoss(weight)
        self.save_path = a.savedir
        os.makedirs(self.save_path, exist_ok=True)

    def eval(self):
        self.discriminator.eval()
        self.generator.eval()
        self.discriminator_pretrained.eval()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            print('********** '+str(self.disc_num)+' **********')
            # print('T04 EER')
            # val_err_04 = self._run_one_epoch(self.test_loader_04)
            # print('Validation EER')
            # val_err_01 = self._run_one_epoch(self.valid_dataset)
            print('T01 EER')
            # val_err_01 = self._run_one_epoch(self.test_loader_01)
            val_err_01 = self.compute_mmd(self.test_loader_01)
            
            # print('T02 EER')
            # val_err_02 = self._run_one_epoch(self.test_loader_02)
            # print('T03 EER')
            # val_err_03 = self._run_one_epoch(self.test_loader_03)

            # print('T04 EER')
            # val_err_04 = self._run_one_epoch_per_fmap(self.test_loader_04, 'T04')
            # print('Validation EER')
            # val_err_01 = self._run_one_epoch_per_fmap(self.valid_dataset, 'Validation')
            # print('T01 EER')
            # val_err_01 = self._run_one_epoch_per_fmap(self.test_loader_01, 'T01')
            # print('T02 EER')
            # val_err_02 = self._run_one_epoch_per_fmap(self.test_loader_02, 'T02')
            # print('T03 EER')
            # val_err_03 = self._run_one_epoch_per_fmap(self.test_loader_03, 'T03')

            # print('Validation EER')
            # val_err_01 = self._run_one_epoch_per_mrd(self.valid_dataset, 'Validation_mrd')
            # print('T01 EER')
            # val_err_01 = self._run_one_epoch_per_mrd(self.test_loader_01, 'T01_mrd')
            # print('T02 EER')
            # val_err_02 = self._run_one_epoch_per_mrd(self.test_loader_02, 'T02_mrd')
            # print('T03 EER')
            # val_err_03 = self._run_one_epoch_per_mrd(self.test_loader_03, 'T03_mrd')
            # print('T04 EER')
            # val_err_04 = self._run_one_epoch_per_mrd(self.test_loader_04, 'T04_mrd')
            
            # print('Validation EER')
            # val_err_01 = self._run_one_epoch_per_band(self.valid_dataset, 'Validation_band')
            # print('T01 EER')
            # val_err_01 = self._run_one_epoch_per_band(self.test_loader_01, 'T01_band')
            # print('T02 EER')
            # val_err_02 = self._run_one_epoch_per_band(self.test_loader_02, 'T02_band')
            # print('T03 EER')
            # val_err_03 = self._run_one_epoch_per_band(self.test_loader_03, 'T03_band')
            # print('T04 EER')
            # val_err_04 = self._run_one_epoch_per_band(self.test_loader_04, 'T04_band')


    def _run_one_epoch(self, data_loader):
        pbar = tqdm(data_loader, total=len(data_loader))
        pred, label = [], []
        target_scores = []
        nontarget_scores = []
        for i, batch in enumerate(pbar):
            sig, key, names, _ = batch
            if '0_T04_139.flac' == names[0] or '1_T04_139.flac' == names[0]: # data error
                continue
            key = key.view(-1).type(torch.int64).to(self.device)
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            
            sig = sig.unsqueeze(1)
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
                    # print(i*(len(d_fake_mrd[0])-1) + j)
                    # print(i,j)
                    mrd_logit_list.append(self.discriminator.discriminators_mrd[i*(len(d_fake_mrd[0])-1) + j](d_real_mrd[i][j].detach(), d_fake_mrd[i][j].detach()))
            
            summed_logit_mrd = sum(mrd_logit_list)
            pred_label = summed_logit_mrd
            
            #pred_label = F.softmax(pred_label, dim=-1)

            batch_score = (pred_label[:, 1]).data.cpu().numpy().ravel()
            key = key.data.cpu().numpy().ravel()
            for i in range(len(key)):
                if key[i] == 1: # spoof
                    target_scores.append(batch_score[i])
                else:
                    nontarget_scores.append(batch_score[i])
            
        #eer, _ = compute_eer(target_scores, nontarget_scores)
        eer = eer_from_scores(target_scores, nontarget_scores)
        print(eer)
        return eer
    
    def _run_one_epoch_per_fmap(self, data_loader, data_name):
        pbar = tqdm(data_loader, total=len(data_loader))
        pred, label = [], []
        target_scores = [[] for _ in range(75)]
        nontarget_scores = [[] for _ in range(75)]
        for i, batch in enumerate(pbar):
            sig, key, names, _ = batch
            if '0_T04_139.flac' == names[0] or '1_T04_139.flac' == names[0]: # data error
                continue
            key = key.view(-1).type(torch.int64).to(self.device)
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            
            sig = sig.unsqueeze(1)
            recons = self.generator(sig, self.sampling_rate)
            recons = recons.detach()
            
            recons = AudioSignal(recons, 16000)
            sig = AudioSignal(sig, 16000)
            d_fake = self.discriminator_pretrained(recons.audio_data)
            d_real = self.discriminator_pretrained(sig.audio_data)
            d_real_mrd = d_real[5:]
            d_fake_mrd = d_fake[5:]
            
            
            mrd_logit_list = []
            for i in range(len(d_fake_mrd)):    # 3
                for j in range(len(d_fake_mrd[0])-1):   # 26-1
                    # print(i*(len(d_fake_mrd[0])-1) + j)
                    # print(i,j)
                    mrd_logit_list.append(self.discriminator.discriminators_mrd[i*(len(d_fake_mrd[0])-1) + j](d_real_mrd[i][j].detach(), d_fake_mrd[i][j].detach()))
            
            key = key.data.cpu().numpy().ravel()
            for i in range(len(mrd_logit_list)):
                pred_label = mrd_logit_list[i]
            
                pred_label = F.softmax(pred_label, dim=-1)

                batch_score = (pred_label[:, 1]).data.cpu().numpy().ravel()
                for k in range(len(key)):
                    if key[k] == 1: # spoof
                        target_scores[i].append(batch_score[k])
                    else:
                        nontarget_scores[i].append(batch_score[k])
        
        with open(data_name+'.txt','w') as f:
            for i in range(len(mrd_logit_list)):  
                eer, _ = compute_eer(target_scores[i], nontarget_scores[i])
                f.write(str(i)+' : '+str(eer)+'\n')
                print(i, eer)
        return eer
    
    def _run_one_epoch_per_mrd(self, data_loader, data_name):
        pbar = tqdm(data_loader, total=len(data_loader))
        pred, label = [], []
        target_scores_1, target_scores_2, target_scores_3 = [], [], []
        nontarget_scores_1, nontarget_scores_2, nontarget_scores_3 = [], [], []
        for i, batch in enumerate(pbar):
            sig, key, names, _ = batch
            if '0_T04_139.flac' == names[0] or '1_T04_139.flac' == names[0]: # data error
                continue
            key = key.view(-1).type(torch.int64).to(self.device)
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            
            sig = sig.unsqueeze(1)
            recons = self.generator(sig, self.sampling_rate)
            recons = recons.detach()
            
            recons = AudioSignal(recons, 16000)
            sig = AudioSignal(sig, 16000)
            d_fake = self.discriminator_pretrained(recons.audio_data)
            d_real = self.discriminator_pretrained(sig.audio_data)
            d_real_mrd = d_real[5:]
            d_fake_mrd = d_fake[5:]
            
            
            mrd_logit_list = []
            for i in range(len(d_fake_mrd)):    # 3
                for j in range(len(d_fake_mrd[0])-1):   # 26-1
                    # print(i*(len(d_fake_mrd[0])-1) + j)
                    # print(i,j)
                    mrd_logit_list.append(self.discriminator.discriminators_mrd[i*(len(d_fake_mrd[0])-1) + j](d_real_mrd[i][j].detach(), d_fake_mrd[i][j].detach()))
            mrd_1_pred_label = sum(mrd_logit_list[:25])
            mrd_2_pred_label = sum(mrd_logit_list[25:50])
            mrd_3_pred_label = sum(mrd_logit_list[50:])
            key = key.data.cpu().numpy().ravel()

            # breakpoint()
            pred_label_1 = F.softmax(mrd_1_pred_label, dim=-1)
            batch_score_1 = (pred_label_1[:, 1]).data.cpu().numpy().ravel()

            pred_label_2 = F.softmax(mrd_2_pred_label, dim=-1)
            batch_score_2 = (pred_label_2[:, 1]).data.cpu().numpy().ravel()
            
            pred_label_3 = F.softmax(mrd_3_pred_label, dim=-1)
            batch_score_3 = (pred_label_3[:, 1]).data.cpu().numpy().ravel()
            for k in range(len(key)):
                if key[k] == 1: # spoof
                    target_scores_1.append(batch_score_1[k])
                    target_scores_2.append(batch_score_2[k])
                    target_scores_3.append(batch_score_3[k])
                else:
                    nontarget_scores_1.append(batch_score_1[k])
                    nontarget_scores_2.append(batch_score_2[k])
                    nontarget_scores_3.append(batch_score_3[k])
        
        with open('analysis/'+data_name+'.txt','w') as f:
            # eer, _ = compute_eer(target_scores_1, nontarget_scores_1)
            eer = eer_from_scores(target_scores_1, nontarget_scores_1)
            f.write('mrd 1 : '+str(eer)+'\n')
            print(eer)

            # eer, _ = compute_eer(target_scores_2, nontarget_scores_2)
            eer = eer_from_scores(target_scores_2, nontarget_scores_2)
            f.write('mrd 2 : '+str(eer)+'\n')
            print(eer)
            
            # eer, _ = compute_eer(target_scores_3, nontarget_scores_3)
            eer = eer_from_scores(target_scores_3, nontarget_scores_3)
            f.write('mrd 3 : '+str(eer)+'\n')
            print(eer)
        return eer
    

    def _run_one_epoch_per_band(self, data_loader, data_name):
        pbar = tqdm(data_loader, total=len(data_loader))
        pred, label = [], []
        target_scores = [[] for _ in range(15)] # 5 bands * 3 mrd
        nontarget_scores = [[] for _ in range(15)]
        pred_label = [0] * 15
        batch_score = [0] * 15
        # target_scores_1, target_scores_2, target_scores_3, target_scores_4, target_scores_5 = [], [], [], [], []
        # nontarget_scores_1, nontarget_scores_2, nontarget_scores_3, nontarget_scores_4, nontarget_scores_5 = [], [], [], [], []
        for i, batch in enumerate(pbar):
            sig, key, names, _ = batch
            if '0_T04_139.flac' == names[0] or '1_T04_139.flac' == names[0]: # data error
                continue
            key = key.view(-1).type(torch.int64).to(self.device)
            key = key.data.cpu().numpy().ravel()
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            
            sig = sig.unsqueeze(1)
            recons = self.generator(sig, self.sampling_rate)
            recons = recons.detach()
            
            recons = AudioSignal(recons, 16000)
            sig = AudioSignal(sig, 16000)
            d_fake = self.discriminator_pretrained(recons.audio_data)
            d_real = self.discriminator_pretrained(sig.audio_data)
            d_real_mrd = d_real[5:]
            d_fake_mrd = d_fake[5:]
            
            
            mrd_logit_list = []
            for i in range(len(d_fake_mrd)):    # 3
                for j in range(len(d_fake_mrd[0])-1):   # 26-1
                    # print(i*(len(d_fake_mrd[0])-1) + j)
                    # print(i,j)
                    mrd_logit_list.append(self.discriminator.discriminators_mrd[i*(len(d_fake_mrd[0])-1) + j](d_real_mrd[i][j].detach(), d_fake_mrd[i][j].detach()))
            
            for i in range(15):
                pred_label = sum(mrd_logit_list[i*5:(i+1)*5])
                pred_label = F.softmax(pred_label, dim=-1)
                batch_score[i] = (pred_label[:, 1]).data.cpu().numpy().ravel()  # 한 샘플에 대해서 15개 logit 나옴.

            for i in range(15):
                for k in range(len(key)):   # 한 배치 안에,
                    if key[k] == 1: # spoof
                        target_scores[i].append(batch_score[i][k])  # 총 117개 샘플이 각 15개 logit씩 저장: T04 기준
                    else:
                        nontarget_scores[i].append(batch_score[i][k])# 총 257개 샘플이 각 15개 logit씩 저장: T04 기준
                    
        # breakpoint()
        with open('analysis/'+data_name+'.txt','w') as f:
            for i in range(15):
                # eer, _ = compute_eer(target_scores[i], nontarget_scores[i])
                eer = eer_from_scores(target_scores[i], nontarget_scores[i])
                f.write(str(i)+' : '+str(eer)+'\n')
                print(eer)
        return eer

    def compute_mmd(self, data_loader):
        pbar = tqdm(data_loader, total=len(data_loader))
        bona_kernel = []
        spoof_kernel = []
        all_counts_bona, all_counts_spoof = [], []
        num = 440   # bona fide 최대 개수.
        cnt_b = cnt_s = 1
        for i, batch in enumerate(pbar):
            sig, key, names, _ = batch
            if '0_T04_139.flac' == names[0] or '1_T04_139.flac' == names[0]: # data error
                continue
            key = key.view(-1).type(torch.int64).to(self.device)
            sig = torch.autograd.Variable(sig.to(self.device, non_blocking=True))
            
            sig = sig.unsqueeze(1)
            
            recons = self.generator(sig, self.sampling_rate)
            sig_z = self.generator.encode(sig, self.sampling_rate)            
            recons_z = self.generator.encode(recons, self.sampling_rate)

            for k in range(len(key)):
                if key[k] == 0:
                    if cnt_b > num:
                        continue
                    data = recons_z[k]
                    data = data.detach().cpu().numpy()
                    np.save('analysis/emb_data/recon_b/recon_b_'+str(cnt_b)+'.npy', data)

                    data = sig_z[k]
                    data = data.detach().cpu().numpy()
                    np.save('analysis/emb_data/gt_b/gt_b_'+str(cnt_b)+'.npy', data)
                    # data = recons[k]
                    # data = data.detach().cpu().numpy()
                    # data = data*32768
                    # write('audio_data/recon/recon_b_'+str(cnt_b)+'.wav', 16000, data[0].astype(np.int16))

                    # data = sig[k]
                    # data = data.detach().cpu().numpy()
                    # data = data*32768
                    # write('audio_data/gt/gt_b_'+str(cnt_b)+'.wav', 16000, data[0].astype(np.int16))
                    cnt_b += 1
                else:
                    if cnt_s > num:
                        continue
                    data = recons_z[k]
                    data = data.detach().cpu().numpy()
                    np.save('analysis/emb_data/recon_s/recon_s_'+str(cnt_s)+'.npy', data)

                    data = sig_z[k]
                    data = data.detach().cpu().numpy()
                    np.save('analysis/emb_data/gt_s/gt_s_'+str(cnt_s)+'.npy', data)
                    # data = recons[k]
                    # data = data.detach().cpu().numpy()
                    # data = data*32768
                    # write('audio_data/recon/recon_s_'+str(cnt_s)+'.wav', 16000, data[0].astype(np.int16))

                    # data = sig[k]
                    # data = data.detach().cpu().numpy()
                    # data = data*32768
                    # write('audio_data/gt/gt_s_'+str(cnt_s)+'.wav', 16000, data[0].astype(np.int16))
                    cnt_s += 1
            if cnt_s > num and cnt_b > num:
                exit()
            
            
            # recons = AudioSignal(recons, 16000)
            # sig = AudioSignal(sig, 16000)
            # d_fake = self.discriminator_pretrained(recons.audio_data)
            # d_real = self.discriminator_pretrained(sig.audio_data)
            # d_real_mrd = d_real[5:]
            # d_fake_mrd = d_fake[5:]
            
            # mrd_logit_list = []
            # batch_size = len(names)
            
            # bins = np.linspace(-1, 1, 201)
            
            # for i in range(len(d_fake_mrd)):
            #     torch.cuda.empty_cache()
            #     for j in range(len(d_fake_mrd[0])-1):
            #         torch.cuda.empty_cache()
            #         logit = self.discriminator.discriminators_mrd[i*(len(d_fake_mrd[0])-1) + j](d_real_mrd[i][j].detach(), d_fake_mrd[i][j].detach())
            #         torch.cuda.empty_cache()
            #         logit_s = F.softmax(logit)
            #         for n in range(batch_size):
            #             if key[n] == 1 and logit_s[n][1]>=0.9 and d_real_mrd[i][j][n].size(-1) < 30:    # memory 터짐.
            #                 _, kernel, _ = self.draw_mmd_histogram(d_real_mrd[i][j][n].detach(), d_fake_mrd[i][j][n].detach())
            #                 torch.cuda.empty_cache()
            #                 # plt.figure()
            #                 # plt.hist(kernel.cpu().numpy(), bins=200)
            #                 # plt.yscale('log')
            #                 # plt.savefig(str(cnt_s)+'_spoof_over_90p.png')
            #                 # torch.cuda.empty_cache()
            #                 # cnt_s += 1
            #                 cnt, _ = np.histogram(kernel.cpu().numpy(), bins=200)
            #                 torch.cuda.empty_cache()
            #                 all_counts_spoof.append(cnt)
                            
            #             # if key[n] == 0 and logit_s[n][0]>=0.9 and d_real_mrd[i][j][n].size(-1) < 30:
            #             #     _, kernel, _ = self.draw_mmd_histogram(d_real_mrd[i][j][n].detach(), d_fake_mrd[i][j][n].detach())
            #             #     torch.cuda.empty_cache()
            #             #     cnt, _ = np.histogram(kernel.cpu().numpy(), bins=200)
            #             #     torch.cuda.empty_cache()
            #             #     all_counts_bona.append(cnt)
            #             #     # torch.cuda.empty_cache()
            #             #     # plt.figure()
            #             #     # plt.hist(kernel.cpu().numpy(), bins=200)
            #             #     # plt.yscale('log')
            #             #     # plt.savefig(str(cnt_b)+'_bona_over_90p.png')
            #             #     # torch.cuda.empty_cache()
            #             #     # cnt_b += 1
            # # if cnt_s >= 10 or cnt_b >= 10:
            # #     exit()
        
            # torch.cuda.empty_cache()
            # if len(all_counts_spoof) >= num or len(all_counts_bona) >= num:
            #     # mean_counts = np.mean(all_counts_bona, axis=0)
            #     # plt.bar((bins[:-1] + bins[1:]) / 2, mean_counts, width=np.diff(bins), align="center", alpha=0.7)
            #     # plt.savefig('bona_10_over_90p.png')
            #     # plt.figure()
            #     # plt.bar((bins[:-1] + bins[1:]) / 2, mean_counts, width=np.diff(bins), align="center", alpha=0.7)
            #     # plt.yscale('log')
            #     # plt.savefig('bona_10_over_90p_log.png')
                
            #     # exit()

            #     mean_counts = np.mean(all_counts_spoof, axis=0)
            #     plt.bar((bins[:-1] + bins[1:]) / 2, mean_counts, width=np.diff(bins), align="center", alpha=0.7)
            #     plt.savefig('spoof_10_over_90p.png')
            #     plt.figure()
            #     plt.bar((bins[:-1] + bins[1:]) / 2, mean_counts, width=np.diff(bins), align="center", alpha=0.7)
            #     plt.yscale('log')
            #     plt.savefig('spoof_10_over_90p_log.png')
            #     breakpoint()
                
            
    
    def draw_mmd_histogram(self, x, y):
        kernel = draw_mmd(x, y)
        # plt.hist(kernel0.mean(dim=0).cpu().numpy(), bins=200)
        kernel_x = kernel.mean(dim=0)
        kernel_y = kernel.mean(dim=1)
        return kernel, kernel_x, kernel_y



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
    parser.add_argument('--checkpoint_path', default="outdir/fmap_mrd_half_ch/chkpt")
    parser.add_argument('--name', default='best_model.ckpt-524_d.pt')
    parser.add_argument('--checkpoint_path_student', default="/home/sam/singfake/outdir/mix_draft/chkpt")
    parser.add_argument('--name_student', default='best_model.ckpt-344_g.pt')
    parser.add_argument('--name_disc', default='best_model.ckpt-344_d.pt')

    parser.add_argument('--summary_interval', default=1000, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument("--train_dir", default=Path('/home/sam/DB/singfake/mixtures'), type=Path)
    parser.add_argument("--test_dir", default=Path('/home/sam/DB/singfake/mixtures'), type=Path)
    parser.add_argument("--training-epochs", metavar="NUM_EPOCHS", default=200,
                            type=int, help="The number of epochs to train. (default: 200)")
    parser.add_argument("--num-workers", default=10, type=int,
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
    

