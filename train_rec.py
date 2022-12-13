import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import torch_utils, helper
from utils.GraphMaker import GraphMaker
from model.trainer import CDSRTrainer
from utils.loader import *
import json
import codecs
import tqdm
import pdb
from fl_devices import Client, Server

def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def arg_parse():
    parser = argparse.ArgumentParser()
    # dataset part
    parser.add_argument('--data_dir', type=str, default='Food-Kitchen', help='Movie-Book, Entertainment-Education')

    # model part
    parser.add_argument('--model', type=str, default="C2DSR", help='model name')
    parser.add_argument('--hidden_units', type=int, default=256, help='lantent dim.')
    parser.add_argument('--num_blocks', type=int, default=2, help='lantent dim.')
    parser.add_argument('--num_heads', type=int, default=1, help='lantent dim.')
    parser.add_argument('--GNN', type=int, default=1, help='GNN depth.')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate.')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                        help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
    parser.add_argument('--lr_decay', type=float, default=1, help='Learning rate decay rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--leakey', type=float, default=0.1)
    parser.add_argument('--maxlen', type=int, default=15)
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--lambda', type=float, default=0.7)

    # train part
    parser.add_argument('--num_iter', type=int, default=100, help='Number of total training iterations.')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size.')
    parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
    parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k iterations.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
    parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
    parser.add_argument('--seed', type=int, default=2040)
    parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
    parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
    parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
    parser.add_argument('--undebug', action='store_false', default=True)
    parser.add_argument('--n_clients', type=int, default=1)
    parser.add_argument('--local_epoch', type=int, default=100, help='Number of local training epochs.')
    parser.add_argument('--eval_interval', type=int, default=1, help='Interval of evalution')
    parser.add_argument('--frac', type=float, default=1, help='Fraction of participating clients')
    args = parser.parse_args()
    return args

def preprocess_dataset(opt):
    n_clients = opt["n_clients"]
    
    train_data = DataLoader(opt['data_dir'], opt['batch_size'], opt, evaluation = -1).data

    valid_data = DataLoader(opt['data_dir'], opt["batch_size"], opt, evaluation = 2).data

    test_data = DataLoader(opt['data_dir'], opt["batch_size"], opt, evaluation = 1).data

    train_data_per_client = len(train_data)//n_clients
    valid_data_per_client = len(valid_data)//n_clients
    test_data_per_client = len(test_data)//n_clients
    client_train_data = [ i for i in range(n_clients)]
    client_valid_data = [ i for i in range(n_clients)]
    client_test_data = [ i for i in range(n_clients)]

    for c_id in range(n_clients):
        if c_id < n_clients - 1:
            client_train_data[c_id] = \
                train_data[c_id * train_data_per_client: (c_id + 1) * train_data_per_client]
            client_valid_data[c_id] = \
                valid_data[c_id * valid_data_per_client: (c_id + 1) * valid_data_per_client]  
            client_test_data[c_id] = \
                test_data[c_id * test_data_per_client: (c_id + 1) * test_data_per_client]  
        else:
            client_train_data[c_id] = train_data[c_id * train_data_per_client: ]
            client_valid_data[c_id] = valid_data[c_id * valid_data_per_client: ]
            client_test_data[c_id] = test_data[c_id * test_data_per_client: ]

    # print(len(client_train_data))
    # print(len(client_train_data[0])) # 804
    # print(len(client_train_data[0][0].shape)) 
    # print(len(client_train_data[1])) # 804
    # print(len(client_train_data[2])) # 804

    import pickle
    with open(os.path.join("dataset", opt['data_dir'], "client_train_data.pkl"), "wb") as f:
        pickle.dump(client_train_data, f)
    with open(os.path.join("dataset", opt['data_dir'], "client_valid_data.pkl"), "wb") as f:
        pickle.dump(client_valid_data, f)
    with open(os.path.join("dataset", opt['data_dir'], "client_test_data.pkl"), "wb") as f:
        pickle.dump(client_test_data, f)

def load_dataset(opt):
    import pickle

    with open(os.path.join("dataset", opt['data_dir'], "client_train_data.pkl"), "rb") as f:
        client_train_data = pickle.load(f)
    with open(os.path.join("dataset", opt['data_dir'], "client_valid_data.pkl"), "rb") as f:
        client_valid_data = pickle.load(f)
    with open(os.path.join("dataset", opt['data_dir'], "client_test_data.pkl"), "rb") as f:
        client_test_data = pickle.load(f)


    # print(len(client_train_data))
    # print(len(client_train_data[0])) # 804
    # print(client_train_data[0][0].shape) 
    # print(len(client_train_data[1])) # 804
    # print(len(client_train_data[2])) # 804

    train_dataloader_list, valid_dataloader_list, test_dataloader_list = [], [], []
    for train_data, valid_data, test_data in zip(client_train_data, client_valid_data, client_test_data):
        train_dataloader = DataLoader.loadfromdata(train_data, opt['data_dir'], opt['batch_size'], opt, evaluation = -1)
        valid_dataloader = DataLoader.loadfromdata(valid_data, opt['data_dir'], opt['batch_size'], opt, evaluation = 2)
        test_dataloader = DataLoader.loadfromdata(test_data, opt['data_dir'], opt['batch_size'], opt, evaluation = 1)

        train_dataloader_list.append(train_dataloader)
        valid_dataloader_list.append(valid_dataloader)
        test_dataloader_list.append(test_dataloader)
    return train_dataloader_list, valid_dataloader_list, test_dataloader_list


def main():
    args = arg_parse()
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)
    init_time = time.time()
    # make opt
    opt = vars(args)

    seed_everything(opt["seed"])

    model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
    model_save_dir = opt['save_dir'] + '/' + model_id
    opt['model_save_dir'] = model_save_dir
    helper.ensure_dir(model_save_dir, verbose=True)
    # save config
    helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
    file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                    header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

    # print model info
    helper.print_config(opt)

    if opt["undebug"]:
        pass
        # opt["cuda"] = False
        # opt["cpu"] = True

    print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))

    preprocess_dataset(opt)

    train_dataloader_list, valid_dataloader_list, test_dataloader_list = load_dataset(opt)


    n_clients = opt["n_clients"]

    print("Data loading done!")

    opt["itemnum"] = opt["source_item_num"] + opt["target_item_num"] + 1


    filename = opt["data_dir"]
    train_data = "./dataset/" + filename + "/traindata_new.txt"
    G = GraphMaker(opt, train_data)
    adj, adj_single = G.adj, G.adj_single
    print("graph loaded!")

    import os

    # os.environ['CUDA_VISIBLE_DEVICES']='0'
    # foo = torch.tensor([1,2,3])
    # foo = foo.to('cuda')

    if opt["cuda"]:
        torch.cuda.empty_cache()
        adj = adj.cuda()
        adj_single = adj_single.cuda()



    # model
    import copy
    if not opt['load']:
        clients = [Client(CDSRTrainer, opt, adj, adj_single, \
            train_dataloader_list[i], valid_dataloader_list[i], test_dataloader_list[i]) for i in range(n_clients)]
        server = Server(CDSRTrainer, opt, adj, adj_single)
    else:
        exit(0)
        
    # 初始化各client的权重
    client_n_samples = [len(client.train_dataloader) for client in clients]
    samples_sum = sum(client_n_samples)
    for client in clients:
        client.weight = len(client.train_dataloader)/samples_sum
            
    global_step = 0
    current_lr = opt["lr"]
    format_str = 'client: {}, {}: step {}/{} (round {}/{}), loss = {:.6f} ({:.3f} sec/epoch), lr: {:.6f}'


    print("Start training:")

    begin_time = time.time()

    # 存储每个client的历史分数
    X_dev_score_history=[[0] for i in range(n_clients)]
    Y_dev_score_history=[[0] for i in range(n_clients)]
    # 存储每个client的最佳测试结果
    X_best_list = [ [0] * 6 for i in range(n_clients)]
    Y_best_list = [ [0] * 6 for i in range(n_clients)]  

    global_step = 0


    num_batch = sum([len(clients[c_id].train_dataloader) for c_id in range(n_clients)]) # 45 * n_clients
    max_steps = opt['num_iter'] * num_batch

    # start training

    # for c_id, train_data
    for round in range(1, opt['num_iter'] + 1):
        avg_val_X_MRR, avg_val_X_NDCG_10, avg_val_X_HR_10, avg_val_Y_MRR, avg_val_Y_NDCG_10, avg_val_Y_HR_10\
            = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        participating_cids = server.select_clients(n_clients, opt['frac']) 
                
        
        if round == 1:
            server.aggregate_weight_updates(clients, participating_cids)
            
        for c_id in participating_cids:
            num_batch = len(clients[c_id].train_dataloader) # 45
            # # max_steps = opt['num_iter'] * num_batch
            # epoch_start_time = time.time()
            # clients[c_id].model.mi_loss = 0

            epoch_start_time = time.time()
            clients[c_id].model.mi_loss = 0      

            global_step, train_loss = clients[c_id].compute_weight_update(global_step, epochs=opt['local_epoch'])
            # clients[c_id].reset()
            
            duration = time.time() - epoch_start_time
            print(format_str.format(c_id, datetime.now(), global_step, max_steps, round, \
                                            opt['num_iter'], train_loss/num_batch, duration, current_lr))
            print("mi:", clients[c_id].trainer.mi_loss/num_batch)
            # torch.save(clients[c_id].trainer.model.state_dict(), "model.json")
        
        server.aggregate_weight_updates(clients, participating_cids)
                
        for c_id in participating_cids:
            clients[c_id].synchronize_with_server(server)   

        if round % opt['eval_interval']:
                continue       

        print("Evaluating on dev set...")
        for c_id in range(n_clients):
            
            if c_id in participating_cids or round == 1:
                val_X_MRR, val_X_NDCG_10, val_X_HR_10, val_Y_MRR, val_Y_NDCG_10, val_Y_HR_10 = clients[c_id].evaluate()
            else: # 如果本轮没参与训练与更新，直接用上一轮的评估结果即可
                val_X_MRR, val_X_NDCG_10, val_X_HR_10, val_Y_MRR, val_Y_NDCG_10, val_Y_HR_10 = clients[c_id].get_old_eval_res()

            avg_val_X_MRR += client.weight * val_X_MRR
            avg_val_X_NDCG_10 += client.weight * val_X_NDCG_10
            avg_val_X_HR_10 += client.weight * val_X_HR_10
            avg_val_Y_MRR += client.weight * val_Y_MRR
            avg_val_Y_NDCG_10 += client.weight * val_Y_NDCG_10
            avg_val_Y_HR_10 += client.weight * val_Y_HR_10
            
            print('client: %d,  val round:%d, time: %f(s), X (MRR: %.4f, NDCG@10: %.4f, HR@10: %.4f), Y (MRR: %.4f, NDCG@10: %.4f, HR@10: %.4f)'
                % (c_id, round, time.time() - begin_time, val_X_MRR, val_X_NDCG_10, val_X_HR_10, val_Y_MRR, val_Y_NDCG_10, val_Y_HR_10))

            if val_X_MRR > max(X_dev_score_history[c_id]) or val_Y_MRR > max(Y_dev_score_history[c_id]):
                if c_id in participating_cids or round == 1:
                    test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_1, test_X_HR_5, test_X_HR_10, \
                        test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10 = clients[c_id].test()
                else:
                    test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_1, test_X_HR_5, test_X_HR_10, \
                        test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10 = clients[c_id].get_old_test_res()

                if val_X_MRR > max(X_dev_score_history[c_id]):
                    # print("X best!")
                    # print([test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_1, test_X_HR_5, test_X_HR_10])
                    X_best_list[c_id] = [test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_1, test_X_HR_5, test_X_HR_10]
                    
                if val_Y_MRR > max(Y_dev_score_history[c_id]):
                    # print("Y best!")
                    # print([test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10])
                    Y_best_list[c_id] = [test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10]

            X_dev_score_history[c_id].append(val_X_MRR)
            Y_dev_score_history[c_id].append(val_Y_MRR)
            
        print('average round:%d, time: %f(s), X (MRR: %.4f, NDCG@10: %.4f, HR@10: %.4f), Y (MRR: %.4f, NDCG@10: %.4f, HR@10: %.4f)'
                % (round, time.time() - begin_time, avg_val_X_MRR, avg_val_X_NDCG_10, avg_val_X_HR_10, avg_val_Y_MRR, avg_val_Y_NDCG_10, avg_val_Y_HR_10))

    X_best_array, Y_best_array = np.array(X_best_list), np.array(X_best_list)
    avg_X_best, avg_Y_best = np.zeros(6), np.zeros(6)
    for c_id in range(n_clients):
        avg_X_best +=  clients[c_id].weight * np.array(X_best_list[c_id])
        avg_Y_best +=  clients[c_id].weight * np.array(Y_best_list[c_id])
    print(avg_X_best)
    print(avg_Y_best)

if __name__ == "__main__":
    main()