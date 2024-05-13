import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
import h5py
import json
import copy
import time
from datetime import datetime

BOARD_SIZE=8

def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions,target=targets)


class MLP(nn.Module):
    def __init__(self, conf):
        super(MLP, self).__init__()  
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        
        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        

    def forward(self, seq):
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0 # to manage earlystopping
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evaluate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evaluate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        
        return best_epoch
    
    
    def evaluate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target,_ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().detach().numpy()
            target=target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep
    
class MLP2(nn.Module):

    def __init__(self, conf):
        super(MLP2, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 64)
        self.lin4 = nn.Linear(64, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        x = self.lin3(x) 
        outp = self.lin4(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep
    

class MLP3(nn.Module):

    def __init__(self, conf):
        super(MLP3, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 64)
        self.lin4 = nn.Linear(64, 32)
        self.lin5 = nn.Linear(32, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        outp = self.lin5(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP4(nn.Module):
    def __init__(self, conf):
        super(MLP4, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep
    

class MLP5(nn.Module):
    def __init__(self, conf):
        super(MLP5, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP6(nn.Module):
    def __init__(self, conf):
        super(MLP6, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep
    

class MLP7(nn.Module): 
    def __init__(self, conf):
        super(MLP7, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep

class MLP7(nn.Module): 
    def __init__(self, conf):
        super(MLP7, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP8(nn.Module):
    def __init__(self, conf):
        super(MLP8, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP9(nn.Module):
    def __init__(self, conf):
        super(MLP9, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP10(nn.Module):
    def __init__(self, conf):
        super(MLP10, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep
    

class MLP11(nn.Module):
    def __init__(self, conf):
        super(MLP11, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP12(nn.Module):
    def __init__(self, conf):
        super(MLP12, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP13(nn.Module):
    def __init__(self, conf):
        super(MLP13, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 640)
        self.lin2 = nn.Linear(640, 640)
        self.lin3 = nn.Linear(640, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP14(nn.Module):
    def __init__(self, conf):
        super(MLP14, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 800)
        self.lin2 = nn.Linear(800, 800)
        self.lin3 = nn.Linear(800, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP15(nn.Module):
    def __init__(self, conf):
        super(MLP15, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 950)
        self.lin2 = nn.Linear(950, 950)
        self.lin3 = nn.Linear(950, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP16(nn.Module):
    def __init__(self, conf):
        super(MLP16, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 1100)
        self.lin2 = nn.Linear(1100, 1100)
        self.lin3 = nn.Linear(1100, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP17(nn.Module):
    def __init__(self, conf):
        super(MLP17, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 1250)
        self.lin2 = nn.Linear(1250, 1250)
        self.lin3 = nn.Linear(1250, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP18(nn.Module):
    def __init__(self, conf):
        super(MLP18, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 1400)
        self.lin2 = nn.Linear(1400, 1400)
        self.lin3 = nn.Linear(1400, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP19(nn.Module):
    def __init__(self, conf):
        super(MLP19, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 1550)
        self.lin2 = nn.Linear(1550, 1550)
        self.lin3 = nn.Linear(1550, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP20(nn.Module):
    def __init__(self, conf):
        super(MLP20, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 1550)
        self.lin2 = nn.Linear(1550, 1550)
        self.lin3 = nn.Linear(1550, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP21(nn.Module):
    def __init__(self, conf):
        super(MLP21, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 1700)
        self.lin2 = nn.Linear(1700, 1700)
        self.lin3 = nn.Linear(1700, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP22(nn.Module):
    def __init__(self, conf):
        super(MLP22, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 1700)
        self.lin2 = nn.Linear(1700, 1700)
        self.lin3 = nn.Linear(1700, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP23(nn.Module):
    def __init__(self, conf):
        super(MLP23, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 1325)
        self.lin2 = nn.Linear(1325, 1325)
        self.lin3 = nn.Linear(1325, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class MLP24(nn.Module):
    def __init__(self, conf):
        super(MLP24, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.lin1 = nn.Linear(self.board_size*self.board_size, 1475)
        self.lin2 = nn.Linear(1475, 1475)
        self.lin3 = nn.Linear(1475, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage earlystopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep
    
class MLP25(nn.Module):
    def __init__(self, conf):
        super(MLP25, self).__init__()  
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        
        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        

    def forward(self, seq):
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0 # to manage earlystopping
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evaluate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evaluate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        
        return best_epoch
    
    
    def evaluate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target,_ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().detach().numpy()
            target=target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep

class MLP26(nn.Module):
    def __init__(self, conf):
        super(MLP26, self).__init__()  
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        
        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        

    def forward(self, seq):
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0 # to manage earlystopping
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evaluate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evaluate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        
        return best_epoch
    
    
    def evaluate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target,_ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().detach().numpy()
            target=target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep

class LSTMs(nn.Module):
    def __init__(self, conf):
        super(LSTMs, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTM/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size, self.hidden_dim,batch_first=True)
        self.hidden2output = nn.Linear(self.hidden_dim*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        outp = self.hidden2output(torch.cat((hn,cn),-1))
        outp = F.softmax(outp, dim=1).squeeze()
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evaluate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evaluate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        
        return best_epoch
    
    
    def evaluate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target_array,lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().clone().detach().numpy()
            target=target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep


class LSTM2(nn.Module):
    def __init__(self, conf):
        super(LSTM2, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size,
                            self.hidden_dim, batch_first=True)
        self.hidden2output = nn.Linear(
            self.hidden_dim*2, self.hidden_dim*2)
        self.hidden2output2 = nn.Linear(
            self.hidden_dim*2, self.board_size*self.board_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):

        seq = np.squeeze(seq)
        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        outp = self.hidden2output(torch.cat((hn, cn), -1))
        outp = self.hidden2output2(outp)
        outp = F.softmax(outp, dim=1).squeeze()
        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class LSTM3(nn.Module):
    def __init__(self, conf):
        super(LSTM3, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size,
                            self.hidden_dim*2, batch_first=True)
        self.hidden2output = nn.Linear(
            self.hidden_dim*4, self.board_size*self.board_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):

        seq = np.squeeze(seq)
        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        outp = self.hidden2output(torch.cat((hn, cn), -1))
        outp = F.softmax(outp, dim=1).squeeze()
        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class LSTM4(nn.Module):
    def __init__(self, conf):
        super(LSTM4, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size,
                            self.hidden_dim, batch_first=True)
        self.hidden2output = nn.Linear(
            self.hidden_dim*2, self.board_size*self.board_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):

        seq = np.squeeze(seq)
        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        outp = self.hidden2output(torch.cat((hn, cn), -1))
        outp = F.softmax(outp, dim=1).squeeze()
        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class LSTM5(nn.Module):
    def __init__(self, conf):
        super(LSTM5, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size,
                            self.hidden_dim, batch_first=True)
        self.hidden2output = nn.Linear(
            self.hidden_dim*2, self.board_size*self.board_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):

        seq = np.squeeze(seq)
        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        outp = self.hidden2output(torch.cat((hn, cn), -1))
        outp = F.softmax(outp, dim=1).squeeze()
        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class LSTM6(nn.Module):
    def __init__(self, conf):
        super(LSTM6, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size,
                            self.hidden_dim, batch_first=True)
        self.hidden2output = nn.Linear(
            self.hidden_dim*2, self.board_size*self.board_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):

        seq = np.squeeze(seq)
        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        outp = self.hidden2output(torch.cat((hn, cn), -1))
        outp = F.softmax(outp, dim=1).squeeze()
        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class LSTM7(nn.Module):
    def __init__(self, conf):
        super(LSTM7, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size,
                            self.hidden_dim*2, batch_first=True)
        self.hidden2output = nn.Linear(
            self.hidden_dim*2, self.board_size*self.board_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):

        seq = np.squeeze(seq)
        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        outp = self.hidden2output(lstm_out[:, -1])
        outp = F.softmax(outp, dim=1).squeeze()
        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class LSTM8(nn.Module):
    def __init__(self, conf):
        super(LSTM8, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size,
                            self.hidden_dim, batch_first=True)
        self.hidden2output = nn.Linear(
            self.hidden_dim, self.hidden_dim)
        self.hidden2output2 = nn.Linear(
            self.hidden_dim, self.board_size*self.board_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):

        seq = np.squeeze(seq)
        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        outp = self.hidden2output(lstm_out[:, -1])
        outp = self.hidden2output2(lstm_out[:, -1])
        outp = F.softmax(outp, dim=1).squeeze()
        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class LSTM9(nn.Module):
    def __init__(self, conf):
        super(LSTM9, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size,
                            self.hidden_dim, batch_first=True)
        self.hidden2output = nn.Linear(
            self.hidden_dim, self.hidden_dim)
        self.hidden2output2 = nn.Linear(
            self.hidden_dim, self.hidden_dim)
        self.hidden2output3 = nn.Linear(
            self.hidden_dim, self.board_size*self.board_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):

        seq = np.squeeze(seq)
        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        outp = self.hidden2output(lstm_out[:, -1])
        outp = self.hidden2output2(lstm_out[:, -1])
        outp = self.hidden2output3(lstm_out[:, -1])
        outp = F.softmax(outp, dim=1).squeeze()
        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class LSTM10(nn.Module):
    def __init__(self, conf):
        super(LSTM10, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size,
                            256, batch_first=True)
        self.hidden2output = nn.Linear(
            256, self.board_size*self.board_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):

        seq = np.squeeze(seq)
        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        outp = self.hidden2output(lstm_out[:, -1])
        outp = F.softmax(outp, dim=1).squeeze()
        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class LSTM11(nn.Module):
    def __init__(self, conf):
        super(LSTM11, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size,
                            64, batch_first=True)
        self.hidden2output = nn.Linear(
            64, self.board_size*self.board_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):

        seq = np.squeeze(seq)
        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        outp = self.hidden2output(lstm_out[:, -1])
        outp = F.softmax(outp, dim=1).squeeze()
        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class LSTM12(nn.Module):
    def __init__(self, conf):
        super(LSTM12, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size,
                            96, batch_first=True)
        self.hidden2output = nn.Linear(
            96, self.board_size*self.board_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):

        seq = np.squeeze(seq)
        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        outp = self.hidden2output(lstm_out[:, -1])
        outp = F.softmax(outp, dim=1).squeeze()
        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class LSTM13(nn.Module):
    def __init__(self, conf):
        super(LSTM13, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size,
                            192, batch_first=True)
        self.hidden2output = nn.Linear(
            192, self.board_size*self.board_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):

        seq = np.squeeze(seq)
        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        outp = self.hidden2output(lstm_out[:, -1])
        outp = F.softmax(outp, dim=1).squeeze()
        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep





class CNN(torch.nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, conf):
        super(CNN, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        
        self.conv_layer1 = torch.nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=(2,2))
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.conv_layer2 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(2,2))
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.relu1 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(16, self.board_size*self.board_size)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.max_pool1(out)

        out = self.conv_layer2(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.relu1(out)
        out = self.fc1(out)
        return F.softmax(out, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class CNN2(torch.nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, conf):
        super(CNN2, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.conv_layer1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2, 2))
        self.conv_layer2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 2))
        self.conv_layer3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 2))
        
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(64, self.board_size*self.board_size)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.conv_layer2(out)
        out = self.relu(out)
        # out = self.max_pool(out)
        
        out = self.conv_layer3(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        return F.softmax(out, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep
            

class CNN3(torch.nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, conf):
        super(CNN3, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.conv_layer1 = torch.nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=(2, 2))
        self.conv_layer2 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(2, 2))
        self.conv_layer3 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(2, 2))
        self.conv_layer4 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(2, 2))
        self.conv_layer5 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(2, 2))

        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(64, self.board_size*self.board_size)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.conv_layer2(out)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.conv_layer3(out)
        out = self.relu(out)
        
        out = self.conv_layer4(out)
        out = self.relu(out)
        
        out = self.conv_layer5(out)
        out = self.relu(out)
        
        out = self.max_pool(out)
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        return F.softmax(out, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class CNN4(torch.nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, conf):
        super(CNN4, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.conv_layer1 = torch.nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(2, 2))
        self.conv_layer2 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(2, 2))
        self.conv_layer3 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(2, 2))
        self.conv_layer4 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(2, 2))
        self.conv_layer5 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(2, 2))

        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(128, self.board_size*self.board_size)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.conv_layer2(out)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.conv_layer3(out)
        out = self.relu(out)

        out = self.conv_layer4(out)
        out = self.relu(out)

        out = self.conv_layer5(out)
        out = self.relu(out)

        out = self.max_pool(out)
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        return F.softmax(out, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class CNN5(torch.nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, conf):
        super(CNN5, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.conv_layer1 = torch.nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(2, 2))
        self.conv_layer2 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(2, 2))
        self.conv_layer3 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(2, 2))
        self.conv_layer4 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(2, 2))
        self.conv_layer5 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(2, 2))
        self.conv_layer6 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(2, 2))
        self.conv_layer7 = torch.nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(2, 2))

        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(256, self.board_size*self.board_size)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.conv_layer2(out)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.conv_layer3(out)
        out = self.relu(out)

        out = self.conv_layer4(out)
        out = self.relu(out)

        out = self.conv_layer5(out)
        out = self.relu(out)
        
        out = self.conv_layer6(out)
        out = self.relu(out)
        
        out = self.conv_layer7(out)
        out = self.relu(out)

        # out = self.max_pool(out)
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        return F.softmax(out, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep
    

class CNN6(torch.nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, conf):
        super(CNN6, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.conv_layer1 = torch.nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(2, 2))
        self.conv_layer2 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(2, 2))
        self.conv_layer3 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(2, 2))
        self.conv_layer4 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(2, 2))
        self.conv_layer5 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(2, 2))
        self.conv_layer6 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(2, 2))
        self.conv_layer7 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(2, 2))
        self.conv_layer8 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(1, 1))
        self.conv_layer9 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(1, 1))

        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(128, self.board_size*self.board_size)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.conv_layer2(out)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.conv_layer3(out)
        out = self.relu(out)

        out = self.conv_layer4(out)
        out = self.relu(out)

        out = self.conv_layer5(out)
        out = self.relu(out)

        out = self.conv_layer6(out)
        out = self.relu(out)

        out = self.conv_layer7(out)
        out = self.relu(out)

        out = self.conv_layer8(out)
        out = self.relu(out)
        
        out = self.conv_layer9(out)
        out = self.relu(out)
        # out = self.max_pool(out)
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        return F.softmax(out, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class CNN7(torch.nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, conf):
        super(CNN7, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.conv_layer1 = torch.nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=(2, 2))
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_layer2 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(2, 2))
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.relu1 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(16, self.board_size*self.board_size)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.max_pool1(out)

        out = self.conv_layer2(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.relu1(out)
        out = self.fc1(out)
        return F.softmax(out, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class CNN8(torch.nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, conf):
        super(CNN8, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.conv_layer1 = torch.nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=(2, 2))
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_layer2 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(2, 2))
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.relu1 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(16, self.board_size*self.board_size)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.max_pool1(out)

        out = self.conv_layer2(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.relu1(out)
        out = self.fc1(out)
        return F.softmax(out, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class CNN9(torch.nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, conf):
        super(CNN9, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.conv_layer1 = torch.nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=(2, 2))
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_layer2 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(2, 2))
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.relu1 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(16, self.board_size*self.board_size)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.max_pool1(out)

        out = self.conv_layer2(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.relu1(out)
        out = self.fc1(out)
        return F.softmax(out, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class CNN10(torch.nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, conf):
        super(CNN10, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.conv_layer1 = torch.nn.Conv2d(
            in_channels=1, out_channels=128, kernel_size=(2, 2))
        self.conv_layer2 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(2, 2))
        self.conv_layer3 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(2, 2))
        self.conv_layer4 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(2, 2))
        self.conv_layer5 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(2, 2))
        self.conv_layer6 = torch.nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(2, 2))
        self.conv_layer7 = torch.nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(2, 2))

        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(256, self.board_size*self.board_size)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.conv_layer2(out)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.conv_layer3(out)
        out = self.relu(out)

        out = self.conv_layer4(out)
        out = self.relu(out)

        out = self.conv_layer5(out)
        out = self.relu(out)

        out = self.conv_layer6(out)
        out = self.relu(out)

        out = self.conv_layer7(out)
        out = self.relu(out)

        # out = self.max_pool(out)
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        return F.softmax(out, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep


class CNN11(torch.nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, conf):
        super(CNN11, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.conv_layer1 = torch.nn.Conv2d(
            in_channels=1, out_channels=256, kernel_size=(2, 2))
        self.conv_layer2 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(2, 2))
        self.conv_layer3 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(2, 2))
        self.conv_layer4 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(2, 2))
        self.conv_layer5 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(2, 2))
        self.conv_layer6 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(2, 2))
        self.conv_layer7 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(2, 2))

        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(256, self.board_size*self.board_size)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.conv_layer2(out)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.conv_layer3(out)
        out = self.relu(out)

        out = self.conv_layer4(out)
        out = self.relu(out)

        out = self.conv_layer5(out)
        out = self.relu(out)

        out = self.conv_layer6(out)
        out = self.relu(out)

        out = self.conv_layer7(out)
        out = self.relu(out)

        # out = self.max_pool(out)
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        return F.softmax(out, dim=-1)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(
                    outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()

            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time()-last_training-start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save +
                           '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print(
                "*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evaluate(dev, device)
        print(
            f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evaluate(self, test_loader, device):

        all_predicts = []
        all_targets = []

        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
#             import pdb
#             pdb.set_trace()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)

        return perf_rep
    
