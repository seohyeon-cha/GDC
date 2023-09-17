from __future__ import print_function
from __future__ import division

import os
import glob

# import libraries
import sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy import stats
import torch
import math
import torch.nn.functional as F
import time
from model import BBGDCGCN
from utils import load_data, accuracy_mrun_np, accuracy, normalize_torch
from conform_pred.GDC_set_predict import conformal_prediction
from conform_pred.GDC_acp_set_predict import ACP
from conform_pred.adaptive_Bay_CP import *

from csv import writer
import csv 
import wandb
import argparse
import ipdb 

def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


parser = argparse.ArgumentParser()
parser.add_argument('--wandb_save', action="store_true")
parser.add_argument('--wandb_proj_name',default='node_classification')
parser.add_argument('--display_name', default='GDC')
parser.add_argument('--seed')
parser.add_argument('--gpu_id', default=0)
parser.add_argument('--wup', default=1)
parser.add_argument('--training', action="store_true")
parser.add_argument('--do_cp', action="store_true")
parser.add_argument('--path')
parser.add_argument('--alpha', default=0.1)
parser.add_argument('--num_cal', default=500)
parser.add_argument('--num_test',default=1000)
parser.add_argument('--nblock',default=2)
parser.add_argument('--num_epochs', default=1700)
parser.add_argument('--num_run', default=12)
parser.add_argument('--dataset', default='citeseer')
parser.add_argument('--do_bayes_cp', action="store_true")
args = parser.parse_args()

if args.wandb_save is True:
        wandb.init(
                project = args.wandb_proj_name,
                name = args.display_name,
                notes = "gdc_original",
            )
if args.gpu_id is not None:
    device = gpu_setup(True, int(args.gpu_id))
if args.wup is not None:
    wup = float(args.wup)
if args.seed is not None:
    seed = int(args.seed)
if args.nblock is not None:
    nblock = int(args.nblock)
if args.num_epochs is not None:
    nepochs = int(args.num_epochs)
if args.num_run is not None:
    num_run = int(args.num_run)
if args.dataset is not None:
    DATASET_NAME = str(args.dataset)

# torch.cuda.device_count()

np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# Load data

adj, features, labels, idx_train, idx_val, idx_test = load_data(DATASET_NAME) # 여기서 dataset 크기 조절하기
adj=adj-sp.diags((adj.diagonal()>0).astype('int'))

# hyper-parameters

nfeat = features.shape[1]
nclass = labels.max().item() + 1
nfeat_list = [nfeat, 128, 128, 128, nclass]
nlay = 4
num_nodes = int(adj.shape[0])
dropout = 0
lr = 0.005 #0.01
weight_decay = 5e-3
mul_type='norm_sec'
con_type='reg'
sym=True

adj = adj + sp.eye(adj.shape[0])
if sym:
    num_edges = int((adj.count_nonzero()+adj.shape[0])/2.0)
else:
    num_edges = adj.count_nonzero()

# finding NZ indecies
if sym:
  adju = sp.triu(adj)
  nz_idx_tupl = adju.nonzero()
  nz_idx_list = []
  for i in range(len(nz_idx_tupl[0])):
    nz_idx_list.append([nz_idx_tupl[0][i], nz_idx_tupl[1][i]])

  nz_idx = torch.LongTensor(nz_idx_list).T
else:
  nz_idx_tupl = adj.nonzero()
  nz_idx_list = []
  for i in range(len(nz_idx_tupl[0])):
    nz_idx_list.append([nz_idx_tupl[0][i], nz_idx_tupl[1][i]])

  nz_idx = torch.LongTensor(nz_idx_list).T


# defining model

model = BBGDCGCN(nfeat_list=nfeat_list
                 , dropout=dropout
                 , nblock=nblock
                 , adj=adj
                 , nlay=nlay
                 , num_edges=num_edges
                 , num_nodes=num_nodes
                 , sym=sym)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Model Summary:")
print(model)
print('----------------------')


adj = torch.FloatTensor(adj.todense())
adj_normt = normalize_torch(adj)

if torch.cuda.is_available():
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    nz_idx = nz_idx.cuda()
    adj = adj.cuda()
    adj_normt = adj_normt.cuda()


labels_np = labels.cpu().numpy().astype(np.int32)
idx_train_np = idx_train.cpu().numpy()
idx_val_np = idx_val.cpu().numpy()
idx_test_np = idx_test.cpu().numpy()



MODEL_NAME='GDC'

def train_pipeline():

    # training
    root_ckpt_dir = './checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(args.gpu_id) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    acc_test_tr = 0

    best_loss = float('inf')
    for epoch in range(nepochs):
        # t = time.time()
        optimizer.zero_grad()
        learnable_T = True
        if learnable_T:
            # activation = nn.Softplus()
            activation = nn.Sigmoid()
            temp = activation(model.temperature) * wup * num_edges / len(idx_train_np)
            print(activation(model.temperature) * wup)
            
        else: 
            temp = wup * num_edges / len(idx_train_np)
        output, kld_loss, nll_loss1, nll_loss2, drop_rates = model(x=features
                                                            , labels=labels
                                                            , adj=adj
                                                            , nz_idx=nz_idx
                                                            , obs_idx=idx_train
                                                            , warm_up=wup
                                                            , adj_normt=adj_normt
                                                            , training=True
                                                            , mul_type=mul_type
                                                            , con_type=con_type
                                                            , fixed_rates=[]
                                                            )
        
        
        # 여기 drop rate 이용해서 wup 만 바꿔주면 됨. 
        index = 0
        l2_reg = None

        block_index = 0
        l2_reg = 0
        for layer in range(model.nlay):
            l2_reg_lay = 0
            if layer==0:
                for name, param in model.gcs[str(block_index)].named_parameters():
                    if 'weight' in name:
                        l2_reg_lay = l2_reg_lay + (param**2).sum()
                block_index += 1
                
            else:
                for iii in range(model.nblock):
                    for name, param in model.gcs[str(block_index)].named_parameters():
                        if 'weight' in name:
                            l2_reg_lay = l2_reg_lay + (param**2).sum()
                    block_index += 1
                    
            l2_reg_lay = (1-drop_rates[layer])/2 * l2_reg_lay
            l2_reg += l2_reg_lay
        
        if epoch < 0: # warming up
            main_loss = nll_loss1 
        else:
            main_loss = nll_loss1 + temp * (kld_loss + l2_reg)
        
            for layy in range(len(model.drpedgs)):
                model.drpedgs[layy].a_uc.grad=1.0 * model.drpedgs[layy].get_derivative_a(torch.sum((model.drpedgs[layy].u_samp-0.5)*(nll_loss2-nll_loss1),dim=0)) + 1.0 * temp * model.drpedgs[layy].get_reg_deriv_a()#without mask
                model.drpedgs[layy].b_uc.grad=1.0 * model.drpedgs[layy].get_derivative_b(torch.sum((model.drpedgs[layy].u_samp-0.5)*(nll_loss2-nll_loss1),dim=0)) + 1.0 * temp * model.drpedgs[layy].get_reg_deriv_b()#without mask
            
        main_loss.backward()
        optimizer.step()

        if main_loss < best_loss:
            best_loss = main_loss

            ckpt_dir = os.path.join(root_ckpt_dir)
            files = glob.glob(ckpt_dir + '/*.pkl')
            for file in files:
                epoch_nb = file.split('_')[-1]
                epoch_nb = int(epoch_nb.split('.')[0])
                if epoch_nb < epoch:
                    os.remove(file)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/best_model_epoch_"+str(epoch)))

        print('Epoch: {:04d}'.format(epoch+1)
            , 'nll: {:.4f}'.format(nll_loss1.item())
            , 'kld: {:.4f}'.format(kld_loss.item()))
        print('----------------------')
        
        # test #
        if epoch % 100 == 0:
            with torch.no_grad():
                model.eval()
                runs_pi = []
                for run in range(num_run):
                    rates = []
                    for layer in range(model.nlay):
                        pi = model.drpedgs[layer].sample_pi()
                        rates.append(pi)
                    runs_pi.append(rates) 

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
                out_runs = torch.stack(outs)
                out_mean = torch.logsumexp(out_runs, dim=0) - np.log(num_run)     
                acc_test_tr = accuracy(out_mean[idx_test], labels[idx_test])

        if args.wandb_save is True:
            wandb.log({'train/_loss': main_loss, 'train/_nll_loss': nll_loss1, 'train/_kld_loss': kld_loss, 'test/_test_acc': acc_test_tr})

    ckpt_dir = os.path.join(root_ckpt_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/final_model_epoch_" + str(epoch)))


def test_CP(path, alpha, num_cal, num_test):

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    total_cp_idx = np.concatenate([idx_val_np, idx_test_np])
    rand_idx = np.random.permutation(len(total_cp_idx))
    idx_cal, idx_test = rand_idx[:num_cal], rand_idx[num_cal:num_cal+num_test]
    idx_cal, idx_test = total_cp_idx[idx_cal], total_cp_idx[idx_test]

    model.load_state_dict(torch.load(path))
    model.eval()
    print(nn.Sigmoid()(model.temperature) * wup)
    test_acc, coverage, inefficiency, bin_dict, qhat = conformal_prediction(model
                                                            , features
                                                            , idx_train_np
                                                            , idx_cal
                                                            , idx_test
                                                            , num_run
                                                            , wup
                                                            , labels
                                                            , nz_idx
                                                            , adj
                                                            , adj_normt
                                                            , alpha)


    
    with open('./CP_out/'+DATASET_NAME+'_GDC_reliability.csv', 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if len(rows) == 0:
        with open('./CP_out/'+DATASET_NAME+'_GDC_reliability.csv', 'w') as f:
            writer_object = csv.writer(f)
            total_samples = 0
            ece = 0
            for bin in range(len(bin_dict)):
                count = bin_dict[bin]['count']

                bin_acc = bin_dict[bin]['bin_acc'] * count
                bin_conf = bin_dict[bin]['bin_conf'] * count
                bin_cov = bin_dict[bin]['bin_cov'] * count
                bin_ineff = bin_dict[bin]['bin_ineff'] * count
                writer_object.writerow([count, bin_acc, bin_conf, bin_cov, bin_ineff])

                # calculate ece
                total_samples += count
                ece += abs(bin_acc-bin_conf)
            ece /= total_samples
            f.close() 
    else:
        total_samples = 0
        ece = 0
        for bin, row in enumerate(rows):

            value = [float(r) for r in row]
            count = bin_dict[bin]['count']

            bin_acc = bin_dict[bin]['bin_acc'] * count
            bin_conf = bin_dict[bin]['bin_conf'] * count
            bin_cov = bin_dict[bin]['bin_cov'] * count
            bin_ineff = bin_dict[bin]['bin_ineff'] * count
            new_value = [count, bin_acc, bin_conf, bin_cov, bin_ineff]
            added_list = [a + b for a, b in zip(value, new_value)]
            rows[bin] = [str(r) for r in added_list]

            # calculate ece
            total_samples += count
            ece += abs(bin_acc-bin_conf)
        ece /= total_samples
        
        with open('./CP_out/'+DATASET_NAME+'_GDC_reliability.csv', 'w') as f:
            for row in rows:
                writer_object = csv.writer(f)
                writer_object.writerow(row)
        f.close()
    
    with open('./CP_out/'+DATASET_NAME+'_GDC_results.csv', 'a') as f:
        writer_object = writer(f)
        writer_object.writerow([DATASET_NAME, wup, num_cal, num_test, test_acc.item(), coverage.item(), inefficiency.item(), qhat, ece])
        f.close()


def multiple_models_CP(model_paths, alpha, num_cal, num_test, mruns):

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    total_cp_idx = np.concatenate([idx_val_np, idx_test_np])
    rand_idx = np.random.permutation(len(total_cp_idx))
    idx_cal, idx_test = rand_idx[:num_cal], rand_idx[num_cal:num_cal+num_test]
    idx_cal, idx_test = total_cp_idx[idx_cal], total_cp_idx[idx_test]


    # gcn model 
    temp_list = []
    model_list = []
    model_dict = {}
    
    with open(model_paths, "r") as file:
        for line in file:
            currentline = line.split(",")
            temp_list.append(float(currentline[0]))
            model = BBGDCGCN(nfeat_list=nfeat_list
                            , dropout=dropout
                            , nblock=nblock
                            , adj=adj
                            , nlay=nlay
                            , num_edges=num_edges
                            , num_nodes=num_nodes
                            , sym=sym)
            model_path = currentline[1].strip()
            model = model.to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model_list.append(model)

    model_dict['wup'] = temp_list
    model_dict['model'] = model_list
    
    coverage, inefficiency = bayesian_QQ(model_dict, features, idx_train, idx_cal, idx_test, mruns, wup, labels, nz_idx, adj, adj_normt, alpha)
    
    print("Inefficiency: {:.4f}".format(inefficiency))
    print("Coverage: {:.4f}".format(coverage))

    with open('./CP_out/'+DATASET_NAME+'_GDC_results.csv', 'a') as f:
        writer_object = writer(f)
        writer_object.writerow([DATASET_NAME, num_cal, num_test, coverage.item(), inefficiency.item()])
        f.close()


def main():

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if args.training==True:
        train_pipeline()
    if args.do_cp == True:
        alpha = float(args.alpha)
        num_cal = int(args.num_cal)
        num_test = int(args.num_test)
        test_CP(args.path, alpha, num_cal, num_test)
    if args.do_bayes_cp==True:
        alpha = float(args.alpha)
        num_cal = int(args.num_cal)
        num_test = int(args.num_test)
        multiple_models_CP(args.path, alpha, num_cal, num_test, num_run)


if __name__ == '__main__':
    main()    
