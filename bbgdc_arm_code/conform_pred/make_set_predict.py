import torch
import torch.nn as nn
import ipdb
import numpy as np
import math
import torch.nn.functional as F
from utils import accuracy_np, accuracy

def conformal_prediction(model, features, idx_train, idx_cal, idx_test, labels, nz_idx, adj, adj_normt, alpha):

    test_acc, set_prediction, bin_dict, test_labels, qhat = validation_CP(model, features, idx_train, idx_cal, idx_test, labels, nz_idx, adj, adj_normt, alpha)
    # test_acc, set_prediction, bin_dict, test_labels, qhat = validation_inductive_CP(model, features, idx_train, idx_cal, idx_test, labels, nz_idx, adj, adj_normt, alpha)
    set_size = torch.sum(set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산
    
    test_size = test_labels.size(0)
    point_prediction = set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return test_acc, coverage, inefficiency, bin_dict, qhat

def validation_inductive_CP(model, features, idx_train, idx_cal, idx_test, labels, nz_idx, adj, adj_normt, alpha):
    

    # x_cal = features[idx_cal].cuda()
    # adj_cal = adj[idx_cal][:, idx_cal].cuda()
    # adj_cal_up = torch.triu(adj_cal)
    # nz_idx_tupl = adj_cal_up.nonzero()
    # nz_idx_cal = nz_idx_tupl.T
    # y_cal = labels[idx_cal].cuda()
    # num_nodes_cal = len(x_cal)
    # num_edges_cal = nz_idx_cal.size(1)

    # with torch.no_grad():
    #     out, _, = model(x=x_cal
    #                             , labels=y_cal
    #                             , adj=adj_cal
    #                             , nz_idx=nz_idx_cal
    #                             , obs_idx=idx_train
    #                             , adj_normt=adj_normt
    #                             , training=False)
        
    #     out = out.detach().cpu() # log-softmax output 
    #     cal_output = out
    #     cal_labels = y_cal.detach().cpu()
    #     n = len(idx_cal)
    #     cal_scores = - cal_output[np.arange(n), cal_labels]
    #     hat_idx = np.ceil((n+1)*(1-alpha))
    #     sorted, idx = torch.sort(cal_scores)
    #     sorted = torch.cat((sorted, torch.Tensor([float('inf')])), -1)
    #     qhat = np.quantile(sorted, 1-alpha, interpolation='higher')

    #     # qhat 구하고 using i,dx_cal
    
    # x_test = features[idx_test].cuda()
    # adj_test = adj[idx_test][:, idx_test].cuda()
    # adj_test_up = torch.triu(adj_test)
    # nz_idx_tupl = adj_test_up.nonzero()
    # nz_idx_test = nz_idx_tupl.T
    # y_test = labels[idx_test].cuda()
    # num_nodes_test = len(x_test)
    # num_edges_test = nz_idx_test.size(1)

    # with torch.no_grad():
    #     out, _, = model(x=x_test
    #                             , labels=y_test
    #                             , adj=adj_test
    #                             , nz_idx=nz_idx_test
    #                             , obs_idx=idx_train
    #                             , adj_normt=adj_normt
    #                             , training=False)
        
    #     acc_test = accuracy(out, y_test).item()
        
    #     out = out.detach().cpu()
    #     test_output = out
    #     test_labels = y_test.detach().cpu()
    #     test_scores = - test_output
    #     set_prediction = test_scores <= qhat

    #     # prediction set

    # bin_dict = reliability_diagram(test_scores, test_labels, qhat)

    # consecutive or simultaneous    
    # idx_cp_cal = np.concatenate([idx_train, idx_cal])
    # x_cal = features[idx_cp_cal].cuda()
    # adj_cal = adj[idx_cp_cal][:, idx_cp_cal].cuda()
    # adj_cal_up = torch.triu(adj_cal)
    # nz_idx_tupl = adj_cal_up.nonzero()
    # nz_idx_cal = nz_idx_tupl.T
    # y_cal = labels[idx_cp_cal].cuda()
    # num_nodes_cal = len(x_cal)
    # num_edges_cal = nz_idx_cal.size(1)

    with torch.no_grad():
        out, _, = model(x=features
                                , labels=labels
                                , adj=adj
                                , nz_idx=nz_idx
                                , obs_idx=idx_train
                                , adj_normt=adj_normt
                                , training=False)
        
        out = out.detach().cpu() # log-softmax output 
        cal_output = out[idx_cal]
        cal_labels = labels[idx_cal].detach().cpu()
        n = len(idx_cal)
        cal_scores = - cal_output[np.arange(n), cal_labels]
        hat_idx = np.ceil((n+1)*(1-alpha))
        sorted, idx = torch.sort(cal_scores)
        sorted = torch.cat((sorted, torch.Tensor([float('inf')])), -1)
        qhat = np.quantile(sorted, 1-alpha, interpolation='higher')

        # qhat 구하고 using i,dx_cal
    
    with torch.no_grad():
        out, _, = model(x=features
                        , labels=labels
                        , adj=adj
                        , nz_idx=nz_idx
                        , obs_idx=idx_train
                        , adj_normt=adj_normt
                        , training=False)

        out_np = out.cpu().data.numpy()
        acc_test = accuracy_np(out_np, labels.cpu().data.numpy(), idx_test)
        out = out.detach().cpu()
        test_output = out[idx_test]
        test_labels = labels[idx_test].detach().cpu()
        test_scores = - test_output
        set_prediction = test_scores <= qhat

        # prediction set

    bin_dict = reliability_diagram(test_scores, test_labels, qhat)
    
    return acc_test, set_prediction, bin_dict, test_labels, qhat


def validation_CP(model, features, idx_train, idx_cal, idx_test, labels, nz_idx, adj, adj_normt, alpha):
    
    with torch.no_grad():
        out, _, = model(x=features
                        , labels=labels
                        , adj=adj
                        , nz_idx=nz_idx
                        , obs_idx=idx_train
                        , adj_normt=adj_normt
                        , training=False)
        out = out.detach().cpu() # log-softmax output 
        cal_output = out[idx_cal]
        cal_labels = labels[idx_cal].detach().cpu()
        n = len(idx_cal)
        cal_scores = - cal_output[np.arange(n), cal_labels]
        hat_idx = np.ceil((n+1)*(1-alpha))
        sorted, idx = torch.sort(cal_scores)
        sorted = torch.cat((sorted, torch.Tensor([float('inf')])), -1)
        qhat = np.quantile(sorted, 1-alpha, interpolation='higher')

        # qhat 구하고 using i,dx_cal
        
    
    with torch.no_grad():
        out, _, = model(x=features
                        , labels=labels
                        , adj=adj
                        , nz_idx=nz_idx
                        , obs_idx=idx_train
                        , adj_normt=adj_normt
                        , training=False)

        out_np = out.cpu().data.numpy()
        acc_test = accuracy_np(out_np, labels.cpu().data.numpy(), idx_test)
        out = out.detach().cpu()
        test_output = out[idx_test]
        test_labels = labels[idx_test].detach().cpu()
        test_scores = - test_output
        set_prediction = test_scores <= qhat

        # prediction set

    bin_dict = reliability_diagram(test_scores, test_labels, qhat)
    
    return acc_test, set_prediction, bin_dict, test_labels, qhat


def reliability_diagram(nlog_sx_scores, test_labels, qhat):

    num_test = nlog_sx_scores.size(0)
    sx_scores = torch.exp(-nlog_sx_scores)
    confidence, prediction = torch.max(sx_scores, dim=1)
    
    # initialize bins
    bin_dict = {}
    num_bins = 20
    for bin_idx in range(num_bins):
        bin_dict[bin_idx] = {}

    for bin_idx in range(num_bins):
        bin_dict[bin_idx]['count'] = 0
        bin_dict[bin_idx]['conf'] = 0
        bin_dict[bin_idx]['acc'] = 0
        bin_dict[bin_idx]['cov'] = 0
        bin_dict[bin_idx]['ineff'] = 0

        bin_dict[bin_idx]['bin_acc'] = 0
        bin_dict[bin_idx]['bin_conf'] = 0
        bin_dict[bin_idx]['bin_cov'] = 0
        bin_dict[bin_idx]['bin_ineff'] = 0

        bin_dict[bin_idx]['corr_count'] = 0
        bin_dict[bin_idx]['incorr_count'] = 0

    # 여기 코드 체크해보기
    for i in range(num_test):


        conf = confidence[i]
        pred = prediction[i]
        label = test_labels[i]
        bin_idx = int(math.ceil((num_bins * conf)-1))
        if bin_idx == -1:
            bin_idx = 0

        # assing to bin with bin_idx
        bin_dict[bin_idx]['count'] += 1
        bin_dict[bin_idx]['conf'] += conf
        bin_dict[bin_idx]['acc'] += (1 if (label==pred) else 0)
        
        if label==pred:
            bin_dict[bin_idx]['corr_count'] += 1
        else:
            bin_dict[bin_idx]['incorr_count'] += 1
            
        # CP
        nc_score = nlog_sx_scores[i]
        set_prediction = nc_score <= qhat
        coverage = (1 if set_prediction[label]==1 else 0)
        ineff = sum(set_prediction).item()
        bin_dict[bin_idx]['cov'] += coverage
        bin_dict[bin_idx]['ineff'] += ineff

    for bin_idx in range(num_bins):
        count = bin_dict[bin_idx]['count']
        if count == 0:
            bin_dict[bin_idx]['bin_acc'] = 0
            bin_dict[bin_idx]['bin_conf'] = 0
            bin_dict[bin_idx]['bin_cov'] = 0
            bin_dict[bin_idx]['bin_ineff'] = 0
        else:    
            bin_dict[bin_idx]['bin_acc'] = float(bin_dict[bin_idx]['acc']/count)
            bin_dict[bin_idx]['bin_conf'] = float(bin_dict[bin_idx]['conf']/count)
            bin_dict[bin_idx]['bin_cov'] = float(bin_dict[bin_idx]['cov']/count)
            bin_dict[bin_idx]['bin_ineff'] = float(bin_dict[bin_idx]['ineff']/count)

    return bin_dict

