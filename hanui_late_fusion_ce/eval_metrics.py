# -*- coding: utf-8 -*-
# @Time    : 2/23/20 1:11 AM
# @Author  : Yuan Gong
# @Affiliation  : University of Notre Dame
# @Email   : yuangongfdu@gmail.com 
# @File    : compute_eer.py

import numpy as np
import sklearn.metrics

# """
# Python compute equal error rate (eer)
# ONLY tested on binary classification

# :param label: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
# :param pred: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
# :param positive_label: the class that is viewed as positive class when computing EER
# :return: equal error rate (EER)
# """
# def compute(label, pred, positive_label=1):
#     # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
#     fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, positive_label)
#     fnr = 1 - tpr

#     # the threshold of fnr == fpr
#     eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

#     # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
#     eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
#     eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

#     # return the mean of eer from fpr and from fnr
#     eer = (eer_1 + eer_2) / 2
#     return eer

import numpy as np
import sys
def compute_det_curve(target_scores, nontarget_scores):
    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]



def _roc_from_scores(target_scores, nontarget_scores):
    s = np.concatenate([target_scores, nontarget_scores])
    y = np.concatenate([np.ones_like(target_scores, dtype=int),
                        np.zeros_like(nontarget_scores, dtype=int)])
    # 점수 내림차순 정렬 (양성일수록 점수 큼 가정)
    idx = np.argsort(-s, kind="mergesort")
    y = y[idx]
    # 누적 TP/FP
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    P = y.sum()
    N = len(y) - P
    # 같은 점수 구간의 마지막 인덱스만 추리기(동률 묶기)
    s_sorted = s[idx]
    _, last_idx = np.unique(s_sorted, return_index=False, return_counts=False, return_inverse=False, axis=None), None
    # 위 한 줄은 파이썬 버전차로 애매하니 아래처럼 처리
    # 고유 점수의 마지막 위치를 얻기
    unique_mask = np.r_[np.diff(s_sorted) != 0, True]
    last_positions = np.where(unique_mask)[0]

    tpr = tp[last_positions] / P
    fpr = fp[last_positions] / N
    # (0,0)과 (1,1) 끝점 추가
    tpr = np.r_[0.0, tpr, 1.0]
    fpr = np.r_[0.0, fpr, 1.0]
    return fpr, tpr

def eer_from_scores(target_scores, nontarget_scores):
    fpr, tpr = _roc_from_scores(target_scores, nontarget_scores)
    far = fpr
    frr = 1 - tpr
    d = far - frr
    # 부호가 바뀌는 구간 찾기
    idx = np.where(np.sign(d[:-1]) * np.sign(d[1:]) <= 0)[0]
    if len(idx) == 0:
        # 교차가 없다면 가장 가까운 점 사용
        k = np.argmin(np.abs(d))
        return 0.5 * (far[k] + frr[k])
    i = idx[0]
    # 선형 보간으로 교차점의 EER 계산
    x0, x1 = far[i], far[i+1]
    y0, y1 = frr[i], frr[i+1]
    # d = far - frr, d0 + t*(d1-d0) = 0 → t = d0/(d0-d1)
    d0, d1 = d[i], d[i+1]
    t = d0 / (d0 - d1 + 1e-12)
    eer = (x0 + t*(x1 - x0) + y0 + t*(y1 - y0)) / 2.0
    return eer

# from sklearn import metrics
# import numpy as np

# # label은 0이 bona fide(진짜), 1이 spoof(가짜)인 binary array
# # score는 시스템이 예측한 점수(일반적으로 spoof에 가까우면 높음)
# def official_eer(label, score):
#     fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
#     fnr = 1 - tpr
#     eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
#     eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
#     print('EER: {:.2f}%'.format(eer * 100))
