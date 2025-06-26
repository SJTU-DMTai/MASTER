import numpy as np
import pandas as pd
import copy
from typing import Optional, List, Tuple, Union, Text
import tqdm
import pprint as pp
import math

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import torch.optim as optim

import qlib
# from qlib.utils import init_instance_by_config
# from qlib.data.dataset import Dataset, DataHandlerLP, DatasetH
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.base import Model
# from qlib.contrib.data.dataset import TSDataSampler
# from qlib.workflow.record_temp import SigAnaRecord, PortAnaRecord
# from qlib.workflow import R, Experiment
# from qlib.workflow.task.utils import TimeAdjuster

import pickle

def zscore(x):
    return (x - x.mean()).div(x.std())

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)  
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def drop_na(x):
    mask = ~torch.isnan(x)
    return mask, x[mask]
    
def dump_data(data, source, stage):
    path = f'data\opensource\{source}_dl_{stage}.pkl'
    print('Dumping data at:', path)
    with open(path, 'wb') as fp: 
        pickle.dump(data, fp)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0,1)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output,  beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output =d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output/self.t, dim=-1)
        return self.d_output*output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z) # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class MASTER(nn.Module):
    def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
                 gate_input_start_index=158, gate_input_end_index=221, beta=None):
        super(MASTER, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.x2y = nn.Linear(d_feat, d_model)
        self.pe = PositionalEncoding(d_model)
        self.tatten = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        self.satten = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        self.temporalatten = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)


    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index] # N, T, D
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        x = self.x2y(src)
        x = self.pe(x)
        x = self.tatten(x)
        x = self.satten(x)
        x = self.temporalatten(x)
        output = self.decoder(x).squeeze(-1)


        return output



class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class MASTERModel(Model):
    def __init__(self, d_feat: int = 158, d_model: int = 256, t_nhead: int = 4, s_nhead: int = 2, gate_input_start_index=158, gate_input_end_index=221,
            T_dropout_rate=0.5, S_dropout_rate=0.5, beta=None, n_epochs = 40, lr = 8e-6, GPU=0, seed=0, train_stop_loss_thred=None, save_path = 'model/', save_prefix= '', benchmark = 'SH000300', market = 'csi300', only_backtest = False):
        
        self.d_model = d_model
        self.d_feat = d_feat

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta

        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        self.benchmark = benchmark
        self.market = market
        self.infer_exp_name = f"{self.market}_MASTER_seed{self.seed}_backtest"

        self.fitted = False
        if self.market == 'csi300':
            self.beta = 5
        else:
            self.beta = 2
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

        self.save_path = save_path
        self.save_prefix = save_prefix
        self.only_backtest = only_backtest

    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def load_model(self, param_path):
        try:
            self.model.load_state_dict(torch.load(param_path, map_location=self.device))
            self.fitted = True
        except:
            raise ValueError("Model not found.") 

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def train_epoch(self, data_loader):
        self.model.train()
        torch.cuda.empty_cache()
        losses = []

        for data in data_loader:
            
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # Additional process on labels
            # Drop extreme labels
            mask, label = drop_extreme(label)
            feature = feature[mask, :, :]
            
            # CSZscoreNorm
            label = zscore(label)

            assert not torch.any(torch.isnan(label))

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # Note the difference: 
            # 1) The qlib.DropnaLabel drop **samples** according to label. 
            # 2) Here we use all samples to compute the inter-stock correlation, but only drop the na labels to compute metrics (loss, etc.).
            mask, label = drop_na(label)
            label = zscore(label)

            pred = self.model(feature.float())
            loss = self.loss_fn(pred[mask], label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = True

    def fit(self, dataset: DatasetH):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        #dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        
        #dump_data(dl_train, self.market, 'train')
        #dump_data(dl_valid, self.market, 'valid')
     
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        #valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=False)

        self.fitted = True
        #best_param = None
        #best_val_loss = 1e3

        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            #print("Epoch %d, train_loss %.4f" % (step, train_loss))
            #val_loss = self.test_epoch(valid_loader)
            #print("Epoch %d, train_loss %.4f, val_loss %.4f " % (step, train_loss,  val_loss))
            #if best_val_loss > val_loss:
                #best_param = copy.deepcopy(self.model.state_dict())
                #best_val_loss = val_loss

            if train_loss <= self.train_stop_loss_thred:
                best_param = copy.deepcopy(self.model.state_dict())
                break

        torch.save(best_param, f'{self.save_path}{self.save_prefix}master_{self.seed}.pkl')

    def predict(self, dataset: DatasetH, use_pretrained = False):
        if use_pretrained:
            self.load_param(f'{self.save_path}{self.save_prefix}master_{self.seed}.pkl')
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        
        #dump_data(dl_test, self.market, 'test')
        
        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        pred_all = []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            pred_all.append(pred.ravel())


        pred_all = pd.DataFrame(np.concatenate(pred_all), index=dl_test.get_index())
        # pred_all = pred_all.loc[self.label_all.index]
        # rec = self.backtest()
        return pred_all
