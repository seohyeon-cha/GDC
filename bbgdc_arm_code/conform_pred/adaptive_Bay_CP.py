import torch
import torch.nn as nn
import ipdb
import numpy as np
import math
import torch.nn.functional as F
from conform_pred.GDC_set_predict import validation_CP
from conform_pred.GDC_acp_set_predict import validation_ACP
from conform_pred.utils import imethod
from conform_pred.class_CP_QQ import calc_matrix_M

def calibration_only(model, features, idx_train, idx_cal, num_run, wup, labels, nz_idx, adj, adj_normt):
    
    mul_type='norm_sec'
    con_type='reg'

    model.eval()
    runs_pi = []

    for run in range(num_run):
        rates = []
        for layer in range(model.nlay):
            pi = model.drpedgs[layer].sample_pi()
            rates.append(pi)
        runs_pi.append(rates)

    with torch.no_grad():
        outs = [None]*num_run
        for j in range(num_run):
            fixed_rates = runs_pi[j]
            outstmp, _, _, _, _ = model(x=features
                                        , labels=labels
                                        , adj=adj
                                        , nz_idx=nz_idx
                                        , obs_idx=idx_train
                                        , warm_up=wup
                                        , adj_normt=adj_normt
                                        , training=False
                                        , mul_type=mul_type
                                        , con_type=con_type
                                        , fixed_rates=fixed_rates
                                                )
            outs[j] = outstmp           

        out_runs = torch.stack(outs).detach().cpu()
        out_mean = torch.logsumexp(out_runs, dim=0) - np.log(num_run)

        cal_output = out_mean[idx_cal].detach().cpu()
        cal_labels = labels[idx_cal].detach().cpu()
        n = len(idx_cal)
        cal_scores = - cal_output[np.arange(n), cal_labels]
    
    return cal_scores, runs_pi


def prediction_only(model, features, idx_train, idx_test, runs_pi, num_run, wup, labels, nz_idx, adj, adj_normt, qhat):

    mul_type='norm_sec'
    con_type='reg'

    with torch.no_grad():
        model.eval()
        outs = [None]*num_run
        for j in range(num_run):
            fixed_rates = runs_pi[j]
            outstmp, _, _, _, _ = model(x=features
                                        , labels=labels
                                        , adj=adj
                                        , nz_idx=nz_idx
                                        , obs_idx=idx_train
                                        , warm_up=wup
                                        , adj_normt=adj_normt
                                        , training=False
                                        , mul_type=mul_type
                                        , con_type=con_type
                                        , fixed_rates=fixed_rates
                                                )
                    
            outs[j] = outstmp       

        out_runs = torch.stack(outs).detach().cpu()
        out_mean = torch.logsumexp(out_runs, dim=0) - np.log(num_run)       

        test_output = out_mean[idx_test]
        test_labels = labels[idx_test].detach().cpu()
        test_scores = - test_output
        set_prediction = test_scores <= qhat

    return set_prediction, test_labels


def bayesian_AGG(model_dict, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    temp_list = model_dict['wup']
    model_list = model_dict['model']
    num_models = len(temp_list)
    bayes_prediction = []
    bayes_size = []
    
    cal_scores_multi_models = []
    runs_multi_models = []
    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        cal_score_per_model, runs_pi = calibration_only(model, features, idx_train, idx_cal, num_run, wup, labels, nz_idx, adj, adj_normt)
        cal_scores_multi_models.append(cal_score_per_model)
        runs_multi_models.append(runs_pi)
    
    # concatenation 
    cal_scores_multi_models = torch.cat(cal_scores_multi_models)
    sorted, idx = torch.sort(cal_scores_multi_models)
    sorted = torch.cat((sorted, torch.Tensor([float('inf')])), -1)
    qhat = np.quantile(sorted, 1-alpha, interpolation='higher')

    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        runs_pi = runs_multi_models[m_idx]
        set_prediction, test_labels = prediction_only(model, features, idx_train, idx_test, runs_pi, num_run, wup, labels, nz_idx, adj, adj_normt, qhat)
        bayes_prediction.append(set_prediction)        
    bayes_prediction = torch.stack(bayes_prediction)

    agg_set_prediction = []
    U = np.random.rand(bayes_prediction.size(1))
    for data_idx in range(bayes_prediction.size(1)):
        sets = bayes_prediction[:, data_idx, :]
        check = torch.sum(sets, dim=0)
        prediction_set = check >= np.ceil(num_models/2 - U[data_idx] + 1/2)
        agg_set_prediction.append(prediction_set)
    
    agg_set_prediction = torch.stack(agg_set_prediction)

    set_size = torch.sum(agg_set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산 
    test_size = test_labels.size(0)
    point_prediction = agg_set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return coverage, inefficiency


def bayesian_AGG_inter(model_dict, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    temp_list = model_dict['wup']
    model_list = model_dict['model']
    num_models = len(temp_list)
    bayes_prediction = []
    bayes_size = []
    
    cal_scores_multi_models = []
    runs_multi_models = []
    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        cal_score_per_model, runs_pi = calibration_only(model, features, idx_train, idx_cal, num_run, wup, labels, nz_idx, adj, adj_normt)
        cal_scores_multi_models.append(cal_score_per_model)
        runs_multi_models.append(runs_pi)
    
    # concatenation 
    cal_scores_multi_models = torch.cat(cal_scores_multi_models)
    sorted, idx = torch.sort(cal_scores_multi_models)
    sorted = torch.cat((sorted, torch.Tensor([float('inf')])), -1)
    qhat = np.quantile(sorted, 1-float(alpha/num_models), interpolation='higher')

    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        runs_pi = runs_multi_models[m_idx]
        set_prediction, test_labels = prediction_only(model, features, idx_train, idx_test, runs_pi, num_run, wup, labels, nz_idx, adj, adj_normt, qhat)
        if m_idx == 0:
            inter_set_prediction = set_prediction
        else:
            inter_set_prediction = torch.bitwise_and(inter_set_prediction, set_prediction)

    set_size = torch.sum(inter_set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산 
    test_size = test_labels.size(0)
    point_prediction = inter_set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return coverage, inefficiency


def bayesian_AGG_adapt(model_dict, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    temp_list = model_dict['wup']
    model_list = model_dict['model']
    num_models = len(temp_list)
    bayes_prediction = []
    bayes_size = []
    
    cal_scores_multi_models = []
    runs_multi_models = []
    qhat_list = []
    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        cal_score_per_model, runs_pi = calibration_only(model, features, idx_train, idx_cal, num_run, wup, labels, nz_idx, adj, adj_normt)
        
        # separate qhat
        sorted, _ = torch.sort(cal_score_per_model)
        sorted = torch.cat((sorted, torch.Tensor([float('inf')])), -1)
        qhat = np.quantile(sorted, 1-alpha, interpolation='higher')

        cal_scores_multi_models.append(cal_score_per_model)
        runs_multi_models.append(runs_pi)
        qhat_list.append(qhat)

    # pick min quantile at each time 
    cal_scores_multi_models = torch.cat(cal_scores_multi_models)
    qhat_list = torch.Tensor(qhat_list)
    min_idx = torch.argmin(qhat_list)
    print(min_idx)

    model = model_list[min_idx]
    runs_pi = runs_multi_models[min_idx]
    set_prediction, test_labels = prediction_only(model, features, idx_train, idx_test, runs_pi, num_run, wup, labels, nz_idx, adj, adj_normt, qhat_list[min_idx])

    set_size = torch.sum(set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산 
    test_size = test_labels.size(0)
    point_prediction = set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return coverage, inefficiency



def bayesian_QQ(model_dict, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    temp_list = model_dict['wup']
    model_list = model_dict['model']
    num_models = len(temp_list)
    bayes_prediction = []
    bayes_size = []
    
    # calculate M matrix
    m = num_models
    n = len(idx_cal)
    M = calc_matrix_M(m, n, .0, mid=False)

    mm = list(np.ravel(M))
    cover_list = [i for i in mm if i > (1-float(alpha/m))]
    if len(cover_list) > 0:
        v = min(cover_list)
    else:
        v = None
        print("No l, k value obtained")
    
    k = int(np.where(M == v)[0])
    l = int(np.where(M == v)[1])
    print(k, ', ', l, ', ', M[k, l])

  
    quantiles = []
    runs_multi_models = []
    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        cal_score_per_model, runs_pi = calibration_only(model, features, idx_train, idx_cal, num_run, wup, labels, nz_idx, adj, adj_normt)
        runs_multi_models.append(runs_pi)
        # separate qhat
        sorted, _ = torch.sort(cal_score_per_model)
        quantiles.append(sorted[l])

    # quantile-of-quantiles 
    quantiles = torch.Tensor(quantiles)
    sorted, _ = torch.sort(quantiles)
    qq = sorted[k]

    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        runs_pi = runs_multi_models[m_idx]
        set_prediction, test_labels = prediction_only(model, features, idx_train, idx_test, runs_pi, num_run, wup, labels, nz_idx, adj, adj_normt, qq)
        bayes_prediction.append(set_prediction)        
    bayes_prediction = torch.stack(bayes_prediction)

    agg_set_prediction = []
    U = np.random.rand(bayes_prediction.size(1))
    for data_idx in range(bayes_prediction.size(1)):
        sets = bayes_prediction[:, data_idx, :]
        check = torch.sum(sets, dim=0)
        prediction_set = check >= np.ceil(num_models/2)
        agg_set_prediction.append(prediction_set)
    
    agg_set_prediction = torch.stack(agg_set_prediction)

    set_size = torch.sum(agg_set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산 
    test_size = test_labels.size(0)
    point_prediction = agg_set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return coverage, inefficiency



def bayesian_MCP(model_dict, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    temp_list = model_dict['wup']
    model_list = model_dict['model']
    num_models = len(temp_list)
    bayes_prediction = []
    bayes_size = []
    
    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        _, set_prediction, _, test_labels, _ = validation_CP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, float(alpha/num_models))
        bayes_prediction.append(set_prediction)

        set_size = torch.sum(set_prediction, dim=1).double()
        bayes_size.append(set_size)
    
    bayes_prediction = torch.stack(bayes_prediction)
    bayes_size = torch.stack(bayes_size)

    min_set_idx = torch.argmin(bayes_size, dim=0)

    min_set_prediction = []
    for (data_idx, idx) in enumerate(min_set_idx):
        min_set = bayes_prediction[idx, data_idx, :]
        min_set_prediction.append(min_set)
    min_set_prediction = torch.stack(min_set_prediction)

    set_size = torch.sum(min_set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산 
    test_size = test_labels.size(0)
    point_prediction = min_set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return coverage, inefficiency

def bayesian_ICP(model_dict, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    temp_list = model_dict['wup']
    model_list = model_dict['model']
    num_models = len(temp_list)
    bayes_prediction = []
    bayes_size = []
    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        _, set_prediction, _, test_labels, _ = validation_CP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, float(alpha/num_models))
        if m_idx == 0:
            inter_set_prediction = set_prediction
        else:
            inter_set_prediction = torch.bitwise_and(inter_set_prediction, set_prediction)

    set_size = torch.sum(inter_set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산 
    test_size = test_labels.size(0)
    point_prediction = inter_set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return coverage, inefficiency

def bayesian_MACP(model_dict, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    temp_list = model_dict['wup']
    model_list = model_dict['model']
    num_models = len(temp_list)
    bayes_prediction = []
    bayes_size = []

    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        _, set_prediction, test_labels, _, _, _, _ = validation_ACP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, float(alpha/num_models))
        bayes_prediction.append(set_prediction)
        set_size = torch.sum(set_prediction, dim=1).double()
        bayes_size.append(set_size)

    bayes_prediction = torch.stack(bayes_prediction)
    bayes_size = torch.stack(bayes_size)

    min_set_idx = torch.argmin(bayes_size, dim=0)

    min_set_prediction = []
    for (data_idx, idx) in enumerate(min_set_idx):
        min_set = bayes_prediction[idx, data_idx, :]
        min_set_prediction.append(min_set)
    min_set_prediction = torch.stack(min_set_prediction)

    set_size = torch.sum(min_set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산 
    test_size = test_labels.size(0)
    point_prediction = min_set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return coverage, inefficiency

def bayesian_IACP(model_dict, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    temp_list = model_dict['wup']
    model_list = model_dict['model']
    num_models = len(temp_list)
    bayes_prediction = []
    bayes_size = []
    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        _, set_prediction, test_labels, _, _, _, _ = validation_ACP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, float(alpha/num_models))

        if m_idx == 0:
            inter_set_prediction = set_prediction
        else:
            inter_set_prediction = torch.bitwise_and(inter_set_prediction, set_prediction)

    set_size = torch.sum(inter_set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산 
    test_size = test_labels.size(0)
    point_prediction = inter_set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return coverage, inefficiency

# majority vote {MVC_CP, MVI_CP, MVC_ACP, MVI_ACP}
def bayesian_MVC_CP(model_dict, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    temp_list = model_dict['wup']
    model_list = model_dict['model']
    num_models = len(temp_list)
    bayes_prediction = []
    bayes_size = []
    epsilon = alpha * np.ceil(num_models/2) / num_models
    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        _, set_prediction, _, test_labels, _ = validation_CP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, float(alpha/num_models))
        bayes_prediction.append(set_prediction)

        set_size = torch.sum(set_prediction, dim=1).double()
        bayes_size.append(set_size)
        
    bayes_prediction = torch.stack(bayes_prediction)
    bayes_size = torch.stack(bayes_size)


    mv_set_prediction = []
    for data_idx in range(bayes_prediction.size(1)):
        sets = bayes_prediction[:, data_idx, :]
        check = torch.sum(sets, dim=0)
        prediction_set = check >= np.ceil(num_models/2)
        mv_set_prediction.append(prediction_set)
    
    mv_set_prediction = torch.stack(mv_set_prediction)

    set_size = torch.sum(mv_set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산 
    test_size = test_labels.size(0)
    point_prediction = mv_set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return coverage, inefficiency

def bayesian_MVI_CP(model_dict, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    temp_list = model_dict['wup']
    model_list = model_dict['model']
    num_models = len(temp_list)
    bayes_prediction = []
    bayes_size = []
    epsilon = imethod(num_models, alpha)
    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        _, set_prediction, _, test_labels, _ = validation_CP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, float(alpha/num_models))
        bayes_prediction.append(set_prediction)

        set_size = torch.sum(set_prediction, dim=1).double()
        bayes_size.append(set_size)
        
    bayes_prediction = torch.stack(bayes_prediction)
    bayes_size = torch.stack(bayes_size)


    mv_set_prediction = []
    for data_idx in range(bayes_prediction.size(1)):
        sets = bayes_prediction[:, data_idx, :]
        check = torch.sum(sets, dim=0)
        prediction_set = check >= np.ceil(num_models/2)
        mv_set_prediction.append(prediction_set)
    
    mv_set_prediction = torch.stack(mv_set_prediction)

    set_size = torch.sum(mv_set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산 
    test_size = test_labels.size(0)
    point_prediction = mv_set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return coverage, inefficiency

def bayesian_MVC_ACP(model_dict, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    temp_list = model_dict['wup']
    model_list = model_dict['model']
    num_models = len(temp_list)
    bayes_prediction = []
    bayes_size = []
    epsilon = alpha * np.ceil(num_models/2) / num_models
    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        _, set_prediction, test_labels, _, _, _, _ = validation_ACP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, float(alpha/num_models))
        bayes_prediction.append(set_prediction)
        set_size = torch.sum(set_prediction, dim=1).double()
        bayes_size.append(set_size)

    bayes_prediction = torch.stack(bayes_prediction)
 
    mv_set_prediction = []
    for data_idx in range(bayes_prediction.size(1)):
        sets = bayes_prediction[:, data_idx, :]
        check = torch.sum(sets, dim=0)
        prediction_set = check >= np.ceil(num_models/2)
        mv_set_prediction.append(prediction_set)
    
    mv_set_prediction = torch.stack(mv_set_prediction)

    # check = torch.sum(bayes_prediction, dim=0)
    # mv_set_prediction = check >= np.ceil(num_models/2)

    set_size = torch.sum(mv_set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산 
    test_size = test_labels.size(0)
    point_prediction = mv_set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return coverage, inefficiency

def bayesian_MVI_ACP(model_dict, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    temp_list = model_dict['wup']
    model_list = model_dict['model']
    num_models = len(temp_list)
    bayes_prediction = []
    bayes_size = []
    epsilon = imethod(num_models, alpha)
    for (m_idx, wup) in enumerate(temp_list):
        model = model_list[m_idx]
        _, set_prediction, test_labels, _, _, _, _ = validation_ACP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, float(alpha/num_models))
        bayes_prediction.append(set_prediction)
        set_size = torch.sum(set_prediction, dim=1).double()
        bayes_size.append(set_size)

    bayes_prediction = torch.stack(bayes_prediction)
 
    mv_set_prediction = []
    for data_idx in range(bayes_prediction.size(1)):
        sets = bayes_prediction[:, data_idx, :]
        check = torch.sum(sets, dim=0)
        prediction_set = check >= np.ceil(num_models/2)
        mv_set_prediction.append(prediction_set)
    
    mv_set_prediction = torch.stack(mv_set_prediction)
    
    set_size = torch.sum(mv_set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산 
    test_size = test_labels.size(0)
    point_prediction = mv_set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return coverage, inefficiency