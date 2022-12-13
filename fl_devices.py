import random
import torch
from torch.utils.data import DataLoader
import numpy as np

def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()
    
def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()
    
def weighted_reduce_add_average(targets, sources):
    for target in targets:
        for name in target:
            target[name].data = torch.sum(torch.stack([source[name].data * weight for (source, weight) in sources]), dim=0).clone()
        
def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


  
class Client(object):
    def __init__(self, model_fn, opt, adj, adj_single, train_dataloader, test_dataloader, val_dataloader):
        self.trainer = model_fn(opt, adj, adj_single)
        self.model = self.trainer.model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader
        
        # 此处为传引用，model参数变化self.W自然变化
        self.W = {key : value for key, value in self.model.named_parameters()}
        
        # self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        # self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        
        # weight for aggregate
        self.weight = 0
        
    def synchronize_with_server(self, server):
        copy(target=self.W, source=server.W)

            
    def compute_weight_update(self, global_step, epochs=1):
        # copy(target=self.W_old, source=self.W)
        # self.optimizer.param_groups[0]["lr"]*=0.99
        # train_stats = train_op(self.model, self.train_loader if not loader else loader, self.optimizer, epochs)
        train_loss = 0
        self.trainer.mi_loss = 0
        for epoch in range(1, epochs + 1):
            train_dataloader = self.train_dataloader
            for batch in train_dataloader:
                # 遍历一次dataloader需要45个batch
                # 一个batch是len18的元组
                # 然后 每个(256, 15)
                loss = self.trainer.train_batch(batch)
                # 计算最后一个epoch的损失并返回
                if epoch == epochs:
                    global_step += 1
                    train_loss += loss
        # subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return global_step, train_loss


    # def reset(self): 
    #     copy(target=self.W, source=self.W_old)
    
    def evaluate(self):
        self.model.eval()
        self.model.graph_convolution()

        self.val_X_pred, self.val_Y_pred = self.get_evaluation_result(self.val_dataloader)
        self.val_X_MRR, self.val_X_NDCG_5, self.val_X_NDCG_10, self.val_X_HR_1, self.val_X_HR_5, self.val_X_HR_10 = self.cal_test_score(self.val_X_pred)
        self.val_Y_MRR, self.val_Y_NDCG_5, self.val_Y_NDCG_10, self.val_Y_HR_1, self.val_Y_HR_5, self.val_Y_HR_10 = self.cal_test_score(self.val_Y_pred)
        # print("")

        return self.val_X_MRR, self.val_X_NDCG_10, self.val_X_HR_10, self.val_Y_MRR, self.val_Y_NDCG_10, self.val_Y_HR_10

    def test(self):
        self.test_X_pred, self.test_Y_pred = self.get_evaluation_result(self.test_dataloader)
        # print("")
        self.test_X_MRR, self.test_X_NDCG_5, self.test_X_NDCG_10, self.test_X_HR_1, self.test_X_HR_5, self.test_X_HR_10 = self.cal_test_score(self.test_X_pred)
        self.test_Y_MRR, self.test_Y_NDCG_5, self.test_Y_NDCG_10, self.test_Y_HR_1, self.test_Y_HR_5, self.test_Y_HR_10 = self.cal_test_score(self.test_Y_pred)

        return self.test_X_MRR, self.test_X_NDCG_5, self.test_X_NDCG_10, self.test_X_HR_1, self.test_X_HR_5, self.test_X_HR_10, \
            self.test_Y_MRR, self.test_Y_NDCG_5, self.test_Y_NDCG_10, self.test_Y_HR_1, self.test_Y_HR_5, self.test_Y_HR_10

    def get_old_eval_res(self):
        return self.val_X_MRR, self.val_X_NDCG_10, self.val_X_HR_10, self.val_Y_MRR, self.val_Y_NDCG_10, self.val_Y_HR_10

    def get_old_test_res(self):
        return self.test_X_MRR, self.test_X_NDCG_5, self.test_X_NDCG_10, self.test_X_HR_1, self.test_X_HR_5, self.test_X_HR_10, \
            self.test_Y_MRR, self.test_Y_NDCG_5, self.test_Y_NDCG_10, self.test_Y_HR_1, self.test_Y_HR_5, self.test_Y_HR_10
    
    @ staticmethod
    def cal_test_score(predictions):
        MRR=0.0
        HR_1 = 0.0
        HR_5 = 0.0
        HR_10 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        valid_entity = 0.0
        # pdb.set_trace()
        for pred in predictions:
            valid_entity += 1
            MRR += 1 / pred
            if pred <= 1:
                HR_1 += 1
            if pred <= 5:
                NDCG_5 += 1 / np.log2(pred + 1)
                HR_5 += 1
            if pred <= 10:
                NDCG_10 += 1 / np.log2(pred + 1)
                HR_10 += 1
            # if valid_entity % 100 == 0:
            #     print('.', end='')
        return MRR/valid_entity, NDCG_5 / valid_entity, NDCG_10 / valid_entity, HR_1 / valid_entity, HR_5 / valid_entity, HR_10 / valid_entity

    def get_evaluation_result(self, evaluation_batch):
        X_pred = []
        Y_pred = []
        for i, batch in enumerate(evaluation_batch):
            X_predictions, Y_predictions = self.trainer.test_batch(batch)
            X_pred = X_pred + X_predictions
            Y_pred = Y_pred + Y_predictions

        return X_pred, Y_pred


class Server(object):
    def __init__(self, model_fn, opt, adj, adj_single):
        self.trainer = model_fn(opt, adj, adj_single)
        self.model = self.trainer.model
        self.W = {key : value for key, value in self.model.named_parameters()}
        self.model_cache = []

    def select_clients(self, n_clients, frac=1.0):
        return sorted(random.sample(range(n_clients), int(n_clients*frac))) 
    
    def aggregate_weight_updates(self, clients, participating_cids):
        #print("reduce 前 server 的权重", "\n", list(self.model.named_parameters()))
        weighted_reduce_add_average(targets=[self.W], sources=[(clients[c_id].W, clients[c_id].weight) for c_id in participating_cids])
        #print("reduce 后 server 的权重", "\n", list(self.model.named_parameters()))
            
            
    # def compute_max_update_norm(self, cluster):
    #     # for client in cluster:
    #     #     print(client.dW)
    #     return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    
    # def compute_mean_update_norm(self, cluster):
    #     return torch.norm(torch.mean(torch.stack([flatten(client.dW) for client in cluster]), 
    #                                  dim=0)).item()

    # def cache_model(self, idcs, params, accuracies):
    #     self.model_cache += [(idcs, 
    #                         {name : params[name].data.clone() for name in params}, 
    #                         [accuracies[i] for i in idcs])]