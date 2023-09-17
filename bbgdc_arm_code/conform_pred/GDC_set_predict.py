import torch
import torch.nn as nn
import ipdb
import numpy as np
import math
import torch.nn.functional as F
from utils import accuracy
from conform_pred.class_CP_QQ import calc_matrix_M

def conformal_prediction(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):

    # test_acc, set_prediction, bin_dict, test_labels, qhat = validation_inductive_CP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha)
    test_acc, set_prediction, bin_dict, test_labels, qhat = validation_CP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha)
    set_size = torch.sum(set_prediction, dim=1).double()
    inefficiency = torch.mean(set_size) # Inefficiency 계산
    
    test_size = test_labels.size(0)
    point_prediction = set_prediction[range(test_size), test_labels]
    coverage = torch.mean(point_prediction.double()) # Coverage 계산

    return test_acc, coverage, inefficiency, bin_dict, qhat


                   
def validation_inductive_CP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    
    x_cal = features[idx_cal].cuda()
    adj_cal = adj[idx_cal][:, idx_cal].cuda()
    adj_cal_up = torch.triu(adj_cal)
    nz_idx_tupl = adj_cal_up.nonzero()
    nz_idx_cal = nz_idx_tupl.T
    y_cal = labels[idx_cal].cuda()
    num_nodes_cal = len(x_cal)
    num_edges_cal = nz_idx_cal.size(1)
    
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

    num_nodes = len(features)
    num_edges = nz_idx.size(1)

    with torch.no_grad():
        outs = [None]*num_run
        for j in range(num_run):
            fixed_rates = runs_pi[j]
            outstmp, kld_loss, nll_loss, _, _ = model(x=x_cal
                                        , labels=y_cal
                                        , adj=adj_cal
                                        , nz_idx=nz_idx_cal
                                        , num_edges=num_edges_cal
                                        , num_nodes=num_nodes_cal
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

        cal_output = out_mean
        cal_labels = y_cal.detach().cpu()
        n = len(idx_cal)
        cal_scores = - cal_output[np.arange(n), cal_labels]
        hat_idx = np.ceil((n+1)*(1-alpha))
        sorted, idx = torch.sort(cal_scores)
        sorted = torch.cat((sorted, torch.Tensor([float('inf')])), -1)
        qhat = np.quantile(sorted, 1-alpha, interpolation='higher')
        print(qhat)
        # qhat 구하고 using i,dx_cal
        
    
    x_test = features[idx_test].cuda()
    adj_test = adj[idx_test][:, idx_test].cuda()
    adj_test_up = torch.triu(adj_test)
    nz_idx_tupl = adj_test_up.nonzero()
    nz_idx_test = nz_idx_tupl.T
    y_test = labels[idx_test].cuda()
    num_nodes_test = len(x_test)
    num_edges_test = nz_idx_test.size(1)

    with torch.no_grad():
        outs = [None]*num_run
        for j in range(num_run):
            fixed_rates = runs_pi[j]
            outstmp, _, _, _, _ = model(x=x_test
                                        , labels=y_test
                                        , adj=adj_test
                                        , nz_idx=nz_idx_test
                                        , num_edges=num_edges_test
                                        , num_nodes=num_nodes_test
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
        acc_test = accuracy(out_mean, y_test)

        test_output = out_mean
        test_labels = y_test.detach().cpu()
        test_scores = - test_output
        set_prediction = test_scores <= qhat

    bin_dict = reliability_diagram(test_scores, test_labels, qhat)

    # # semi-inductive
    # mul_type='norm_sec'
    # con_type='reg'

    # model.eval()
    # runs_pi = []
    # for run in range(num_run):
    #     rates = []
    #     for layer in range(model.nlay):
    #         pi = model.drpedgs[layer].sample_pi()
    #         rates.append(pi)
    #     runs_pi.append(rates)

    # num_nodes = len(features)
    # num_edges = nz_idx.size(1)

    # with torch.no_grad():
    #     outs = [None]*num_run
    #     for j in range(num_run):
    #         fixed_rates = runs_pi[j]
    #         outstmp, _, _, _, _ = model(x=features
    #                                     , labels=labels
    #                                     , adj=adj
    #                                     , nz_idx=nz_idx
    #                                     , num_edges=num_edges
    #                                     , num_nodes=num_nodes
    #                                     , obs_idx=idx_train
    #                                     , warm_up=wup
    #                                     , adj_normt=adj_normt
    #                                     , training=False
    #                                     , mul_type=mul_type
    #                                     , con_type=con_type
    #                                     , fixed_rates=fixed_rates
    #                                     )
            
    #         outs[j] = outstmp    

    #     out_runs = torch.stack(outs).detach().cpu()
    #     out_mean = torch.logsumexp(out_runs, dim=0) - np.log(num_run)

    #     cal_output = out_mean[idx_cal]
    #     cal_labels = labels[idx_cal].detach().cpu()
    #     n = len(idx_cal)
    #     cal_scores = - cal_output[np.arange(n), cal_labels]
    #     hat_idx = np.ceil((n+1)*(1-alpha))
    #     sorted, idx = torch.sort(cal_scores)
    #     sorted = torch.cat((sorted, torch.Tensor([float('inf')])), -1)
    #     qhat = np.quantile(sorted, 1-alpha, interpolation='higher')
    #     print(qhat)
    #     # qhat 구하고 using i,dx_cal
        
    
    # with torch.no_grad():
    #     outs = [None]*num_run
    #     for j in range(num_run):
    #         fixed_rates = runs_pi[j]
    #         outstmp, _, _, _, _ = model(x=features
    #                                     , labels=labels
    #                                     , adj=adj
    #                                     , nz_idx=nz_idx
    #                                     , num_edges=num_edges
    #                                     , num_nodes=num_nodes
    #                                     , obs_idx=idx_train
    #                                     , warm_up=wup
    #                                     , adj_normt=adj_normt
    #                                     , training=False
    #                                     , mul_type=mul_type
    #                                     , con_type=con_type
    #                                     , fixed_rates=fixed_rates
    #                                     )
            
    #         outs[j] = outstmp       

    #     out_runs = torch.stack(outs).detach().cpu()
    #     out_mean = torch.logsumexp(out_runs, dim=0) - np.log(num_run)       
    #     acc_test = accuracy(out_mean[idx_test], labels[idx_test])

    #     test_output = out_mean[idx_test]
    #     test_labels = labels[idx_test].detach().cpu()
    #     test_scores = - test_output
    #     set_prediction = test_scores <= qhat

    # bin_dict = reliability_diagram(test_scores, test_labels, qhat)
    
    return acc_test, set_prediction, bin_dict, test_labels, qhat



def validation_CP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    
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

        cal_output = out_mean[idx_cal]
        cal_labels = labels[idx_cal].detach().cpu()
        n = len(idx_cal)
        cal_scores = - cal_output[np.arange(n), cal_labels]
        hat_idx = np.ceil((n+1)*(1-alpha))
        sorted, idx = torch.sort(cal_scores)
        sorted = torch.cat((sorted, torch.Tensor([float('inf')])), -1)
        qhat = np.quantile(sorted, 1-alpha, interpolation='higher')
        print(qhat)
        # qhat 구하고 using i,dx_cal
        

    
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
        acc_test = accuracy(out_mean[idx_test], labels[idx_test])

        test_output = out_mean[idx_test]
        test_labels = labels[idx_test].detach().cpu()
        test_scores = - test_output
        set_prediction = test_scores <= qhat

    bin_dict = reliability_diagram(test_scores, test_labels, qhat)
    
    return acc_test, set_prediction, bin_dict, test_labels, qhat

def validation_QQ_CP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    
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

    #     # calculate M matrix
    # m = num_run
    # n = len(idx_cal)
    # M = calc_matrix_M(m, n, .0, mid=False)

    # mm = list(np.ravel(M))
    # cover_list = [i for i in mm if i > (1-alpha)]
    # if len(cover_list) > 0:
    #     v = min(cover_list)
    # else:
    #     v = None
    #     print("No l, k value obtained")
    
    # k = int(np.where(M == v)[0])
    # l = int(np.where(M == v)[1])
    # print(k, ', ', l, ', ', M[k, l])
    k = 12
    l = 44

    quantiles = []
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
            output = F.log_softmax(outstmp, dim=1)
            cal_output = output[idx_cal].detach().cpu()
            cal_labels = labels[idx_cal].detach().cpu()
            n = len(idx_cal)
            cal_scores = - cal_output[np.arange(n), cal_labels]
            sorted, _ = torch.sort(cal_scores)
            sorted = torch.cat((sorted, torch.Tensor([float('inf')])), -1)
            quantile = sorted[l-1]
            quantiles.append(quantile)

        out_runs = torch.stack(outs).detach().cpu()
    
    quantiles = torch.Tensor(quantiles)
    sorted, idx = torch.sort(quantiles)
    sorted = torch.cat((sorted, torch.Tensor([float('inf')])), -1)
    qhat = sorted[k-1]
  
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
        acc_test = accuracy(out_mean[idx_test], labels[idx_test])

        test_output = out_mean[idx_test]
        test_labels = labels[idx_test].detach().cpu()
        test_scores = - test_output
        set_prediction = test_scores <= qhat

    bin_dict = reliability_diagram(test_scores, test_labels, qhat)
    
    return acc_test, set_prediction, bin_dict, test_labels, qhat



def validation_Bayes_CP(model, features, idx_train, idx_cal, idx_test, num_run, wup, labels, nz_idx, adj, adj_normt, alpha):
    
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
        mrun_cal_scores = []
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
            output = F.log_softmax(outstmp, dim=1)
            cal_output = output[idx_cal].detach().cpu()
            cal_labels = labels[idx_cal].detach().cpu()
            n = len(idx_cal)
            cal_scores = - cal_output[np.arange(n), cal_labels]
            mrun_cal_scores.append(cal_scores)
        
        mrun_cal_scores = torch.cat(mrun_cal_scores)
        sorted, idx = torch.sort(mrun_cal_scores)
        sorted = torch.cat((sorted, torch.Tensor([float('inf')])), -1)
        qhat = np.quantile(sorted, 1-alpha, interpolation='higher')
        out_runs = torch.stack(outs).detach().cpu()
        print(qhat)
        # qhat 구하고 using i,dx_cal
        
    
    with torch.no_grad():
        model.eval()
        mrun_outputs = []
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
            output = F.log_softmax(outstmp, dim=1)
            test_output = output[idx_test]
            mrun_outputs.append(test_output)

        mrun_outputs = torch.stack(mrun_outputs).detach().cpu()
        
        out_runs = torch.stack(outs).detach().cpu()
        out_mean = torch.logsumexp(out_runs, dim=0) - np.log(num_run)       
        acc_test = accuracy(out_mean[idx_test], labels[idx_test])

        test_output = out_mean[idx_test]
        test_labels = labels[idx_test].detach().cpu()
        test_scores = - test_output
        # set_prediction = test_scores <= qhat

    bin_dict = reliability_diagram(test_scores, test_labels, qhat)
    
    # new 
    mrun_agg_count = (-mrun_outputs) <= qhat
    agg_count = torch.sum(mrun_agg_count, 0)
    u = np.random.rand(agg_count.size(0), 1)
    set_prediction = agg_count >= torch.tensor(np.ceil((num_run+1)/2 - u))
    
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

