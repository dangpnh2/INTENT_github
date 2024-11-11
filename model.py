import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from types import SimpleNamespace
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from io import open
import torch
from collections import Counter

import numpy as np

from typing import Tuple, Dict, Union, List

from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel


# In[2]:



class SentSeq(nn.Module):
    def __init__(self, config, hidden_size, num_intents, max_word_sent, embed_size, n_layers=1, dropout=0, device='cpu', ):
        super(SentSeq, self).__init__()
        self.config = config
        self.device = device
        self.categorical_dims_sent = 2
        self.categorical_dims_doc = num_intents
        self.max_word_sent = max_word_sent

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
#         self.embedding = nn.Embedding(voc.num_words, 300)
        
        
        self.bert = BertModel.from_pretrained(self.config.bert_model_name)
#         for p in self.bert.parameters():
#             p.requires_grad = False
        self.gru = nn.GRU(input_size=self.config.embed_size, hidden_size=self.hidden_size*2)
        
        self.rnn = nn.RNN(input_size=self.config.embed_size, hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.config.num_intents)
        
        # topic feat
        self.use_topic = True
    
    
    
    def forward(self, input_ids_bert, attention_mask_bert): 
        
        ## for L
        N = input_ids_bert.shape[0]
        
    
        outputs = self.bert(input_ids=input_ids_bert, attention_mask=attention_mask_bert)
        last_hidden_state  = outputs[0] # las_hidden; Nxmax_wordsxhidden
        out_h_docs = torch.zeros((N, self.config.max_sent_num, 768)).to(self.config.device)
        total_sents = []
        
       
        for doc_idx in range(N):
            mask = input_ids_bert[doc_idx] == 103 ## SEP token scibert(103); biobert(102)
            mask_cls = input_ids_bert[doc_idx]==102 ## CLS token scibert (102); biobert(101)
            out_each_sent = last_hidden_state[doc_idx][mask] 
            cls_emb = last_hidden_state[doc_idx][mask_cls] 
            total_sent = mask.sum()
            total_sents.append(total_sent)
            out_each_sent = out_each_sent.reshape(-1, self.config.embed_size) # total_sents, feature_size

            ## add start token to doc
            final_feat = torch.cat([cls_emb, out_each_sent], 0)
            out_h_docs[doc_idx,:total_sent,:] = final_feat[:-1,:].squeeze(0)
            
        return out_h_docs
    


# In[15]:


def masked_softmax(vec, mask, dim=-1, epsilon=1e-20):
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps/masked_sums)

def masked_softmax_logsumexp(vec, mask, dim=-1, epsilon=1e-20):
#     exps = torch.exp(vec)
#     masked_exps = exps * mask.float()
#     masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    masked_exps = vec * mask.float() + (1-mask.float())*(-1e30)
#     masked_exps = vec + (mask.float()+epsilon).log()
    
    return torch.exp(masked_exps - torch.logsumexp(masked_exps, dim, keepdim=True)) ## TODO: Check nan


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """

    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y#y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    
    return y_hard#.view(-1, latent_dim * categorical_dim)



class VAEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.num_input, self.h1, self.h2, self.num_topic, self.drop_rate = num_input, h1, h2, num_topic, drop_rate
        self.config = config
        self.fc1 = nn.Linear(self.config.num_input, self.config.h1)

        self.fc2 = nn.Linear(self.config.h1, self.config.h2)
        self.fc2_dropout = nn.Dropout(0.2)
        
#         self.theta_drop     = nn.Dropout(self.config.drop_rate)

        self.mean_fc    = nn.Linear(self.config.h2, self.config.num_topics)   
        self.mean_bn    = nn.BatchNorm1d(self.config.num_topics)              # bn for mean
        self.logvar_fc  = nn.Linear(self.config.h2, self.config.num_topics)        # 100  -> 2
        self.logvar_bn  = nn.BatchNorm1d(self.config.num_topics)     
        
        prior_mean   = torch.Tensor(1, self.config.num_topics).fill_(0)
        prior_var    = torch.Tensor(1, self.config.num_topics).fill_(1)
        self.prior_mean = nn.Parameter(prior_mean, requires_grad=False)
        self.prior_var  = nn.Parameter(prior_var, requires_grad=False)
        self.prior_logvar = nn.Parameter(prior_var.log(), requires_grad=False)
        
    def forward(self, input_):
        
        h1 = F.softplus(self.fc1(input_))#F.tanh(self.fc1(input_))
        h2 = F.softplus(self.fc2(h1))#F.tanh(self.fc2(h1))
        h2 = self.fc2_dropout(h2)
        

        self.posterior_mean = self.mean_bn(self.mean_fc (h2)) 
        self.posterior_logvar = self.logvar_bn(self.logvar_fc(h2)) 
        self.posterior_var = self.posterior_logvar.exp()

        eps = input_.data.new().resize_as_(self.posterior_mean .data).normal_(std=1) # noise
        self.theta = self.posterior_mean  + self.posterior_var .sqrt() * eps  # NxT
        self.theta = torch.softmax(self.theta, dim=-1)
        #self.theta = self.theta_drop(self.theta)
        return self.theta
    
    def compute_KL(self):
        prior_mean   = self.prior_mean.expand_as(self.posterior_mean)
        prior_var    = self.prior_var.expand_as(self.posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(self.posterior_mean)
        var_division    = self.posterior_var  / prior_var #Nx2
        diff            = self.posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - self.posterior_logvar
        KL = 0.5 * ( (var_division + diff_term + logvar_division).sum(-1) - self.config.num_topics) 
        return KL
    
class VAEOneTheta(nn.Module):
    def __init__(self, config, V):
        super().__init__()
        self.config = config
        # V
        self.bow_embeddings = nn.Embedding.from_pretrained(V, freeze=False, padding_idx=0)#nn.Embedding(config.num_input, 768, padding_idx=0)#
        # T
        self.topic_embeddings = nn.init.normal_(nn.Parameter(torch.Tensor(self.config.num_topics, self.config.embed_size)), 0, 0.01)
        
        # Beta
        self.intent_proportion = nn.init.normal_(nn.Parameter(torch.Tensor(self.config.num_intents, self.config.vocab_size)), 0, 0.01)
#         self.intent_embeddings = nn.init.normal_(nn.Parameter(torch.Tensor(self.config.num_intents, self.config.embed_size)), 0, 0.01)
        
        self.psi_bn    = nn.BatchNorm1d(self.config.vocab_size, affine=True, eps=1e-05, momentum=0.1)   
        self.topic_embeddings_bn    = nn.BatchNorm1d(self.config.embed_size)   
        
        self.beta_bn    = nn.BatchNorm1d(self.config.vocab_size)   
        self.model_enc = VAEMLP(self.config)#.to(self.config.device).float()
        #self.fc_b = nn.Linear(self.config.embed_size, 2)
        self.fc_b1 = nn.Linear(self.config.embed_size, self.config.hidden_size)#self.config.embed_size
        self.fc_b2 = nn.Linear(self.config.hidden_size, 2)
        self.alpha = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        
        #self.fc_z = nn.Linear(self.config.embed_size, self.config.num_topics)
        self.fc_z1 = nn.Linear(self.config.embed_size, self.config.hidden_size)
        self.fc_z2 = nn.Linear(self.config.hidden_size, self.config.num_topics)
        
        #self.initS = nn.Parameter(torch.randn(1, self.config.embed_size))
        #self.fc_weighted = nn.Linear(self.config.embed_size*2, self.config.embed_size)
        self.lstm = nn.LSTM(input_size=2*self.config.embed_size, hidden_size=self.config.hidden_size, dropout=0.6)
        self.gru = nn.GRU(input_size=2*self.config.embed_size, hidden_size=self.config.hidden_size, dropout=0.6)
        
        self.fc_logits = nn.Linear(self.config.hidden_size, self.config.num_intents)
        self.drop_fc   = nn.Dropout(0.0)
        self.drop_fc_attention   = nn.Dropout(0.6)
        
        self.u_w  = nn.Parameter(torch.Tensor(1, self.config.embed_size))
        self.bernoulii_p = 0.5
        
        
        
        
        
        self.bias_topic_words = nn.init.normal_(nn.Parameter(torch.Tensor(self.config.num_topics, self.config.vocab_size)), 0, 0.01)
        self.beta_bn    = nn.BatchNorm1d(self.config.vocab_size, affine=True)   
        self.initTs = nn.Parameter(torch.randn(1, self.config.embed_size)*0.01)
        self.fc_weighted2 = nn.Linear(self.config.embed_size, self.config.hidden_size, bias=False)
        self.tscore_i_fc = nn.Linear(self.config.hidden_size, 1, bias=False)
        
        if self.config.use_topic==1:
            self.fc_weighted = nn.Linear(self.config.embed_size, self.config.hidden_size, bias=False)
            self.bias_ui = nn.Parameter(torch.randn(1, self.config.hidden_size)*0.01)
            self.fc_sep1 = nn.Linear(self.config.embed_size*2, 300)#self.config.embed_size*2
            self.fc_sep2 = nn.Linear(300, self.config.hidden_size)
            self.fc_sep3 = nn.Linear(200, self.config.hidden_size)
            
            self.fc_bert_feature = nn.Linear(self.config.embed_size, 300)
            self.fc_topic_feature = nn.Linear(self.config.embed_size, 300)
            self.weight_ts = nn.Parameter(torch.randn(1, self.config.embed_size)*0.01)
        else:
#             self.fc_weighted = nn.Linear(self.config.embed_size, self.config.hidden_size)
            self.fc_sep1 = nn.Linear(self.config.embed_size, 300)
            self.fc_sep2 = nn.Linear(300, self.config.hidden_size)
            self.fc_sep3 = nn.Linear(self.config.embed_size*2, self.config.hidden_size)
        
    def compute_recon(self, theta):
        topic_words = torch.matmul((self.topic_embeddings), self.bow_embeddings.weight.T)
        topic_words = torch.softmax(topic_words, dim=-1)
        
        recon = torch.matmul(theta, topic_words) #NxW 14,28,4981
        
        return recon
    
    def compute_KL_def_prior(self, logits_posterior):
        q_ = torch.softmax(logits_posterior, dim=-1)
        
#         log_ratio = torch.log(q_ / torch.sigmoid(self.alpha) + 1e-20) ## TODO: FIX 0.5->learnable param
        log_ratio = torch.log(q_ * 2 + 1e-20) ## TODO: FIX 0.5->learnable param
        KLD = torch.sum(q_ * log_ratio, dim=-1)
        return KLD
    
    def compute_KL(self, prior, logits_posterior):
        q_ = torch.softmax(logits_posterior, dim=-1)
        p_ = prior#torch.softmax(logits_prior, dim=-1)
        log_ratio = torch.log(q_ / p_ + 1e-20)
        KLD = torch.sum(q_ * log_ratio, dim=-1)
        return KLD
    
    def compute_psi(self):
        
#         topic_words = self.psi_bn (torch.matmul(self.topic_embeddings, self.bow_embeddings.weight.T))
        topic_words = torch.matmul((self.topic_embeddings), self.bow_embeddings.weight.T)
        topic_words = (topic_words)#self.psi_bn
        topic_words = torch.softmax(topic_words, dim=-1) # TxV

#         pw = (z * topic_words.T[input_index]).sum(-1) # -> torch.Size([8, 26, 71])
        return topic_words
    
        
    def forward(self, input_words, input_ids, input_w_mask, input_sent_mask, h_sents, sent_groundtruth_batch, train=True, compute_perp=False):
        
        
        ## ============Draw theta=====
        KL_loss_MLP_arr = []
        input_one_hot = torch.nn.functional.one_hot(input_words, num_classes=self.config.vocab_size).to(self.config.device) #BS, sent, words, V
        input_doc_ = torch.sum(input_one_hot, (1,2)).float()
        input_doc_[:, 0:1] = 0 # ignore padding and SOS
        #input_doc_ = input_doc_[:, ]
        
        theta = self.model_enc(input_doc_).float() #N,T
        KL_loss_MLP_arr = self.model_enc.compute_KL()
        
        psi = self.compute_psi() #T,V
        ## =======Generate b========
        embedded = self.bow_embeddings(input_words) #[16, 31, 137, 768]
        b_list = []
        z_list = []
        h_sents_posterior = []
        h_sents_prior = []
        KL_gumbel_b=[]
        KL_gumbel_z=[]
        N, max_sents, _ = h_sents.shape

#         ## ============Draw b===========
#         for sent in range(embedded.shape[1]): 
#             logits_posterior_b = (self.fc_b2(torch.relu(self.fc_b1(embedded[:,sent,:,:])))) #[BS, sents, 2]
            
# #             b = F.gumbel_softmax(logits_posterior_b, tau=0.1, hard=False) # b: 14x73x2 # [0,1]: topic; [1,0]: intent
#             b = gumbel_softmax(logits_posterior_b, 0.1, False)
#             b_list.append(b)#.squeeze()
#             KL_word = self.compute_KL_def_prior(logits_posterior_b)
#             KL_gumbel_b.append(KL_word)
            
        ## ============Compute p(z|wd)===========     
        
#         theta # N,T
#         psi[:, input_words] #[T, N, sents, words]
                
#         b_list = torch.stack(b_list, dim=1)
        
#         word_indicator = b_list * input_w_mask.unsqueeze(3).repeat(1,1,1,2)
        
#         KL_gumbel_b = torch.stack(KL_gumbel_b, dim=1) 

        #====================== compute L ===================
        out_rnn_docs = torch.zeros((N, self.config.max_sent_num, self.config.hidden_size)).to(self.config.device)
        
        
        for doc_idx in range(N):
            topic_feature_sents = torch.zeros((self.config.max_sent_num, self.config.embed_size)).to(self.config.device)
            S_sents = []
            mask = input_ids[doc_idx] == 103 ## SEP token
            total_sents = mask.sum()
            
            for sent_idx in range(total_sents):
                prev_sent_idx = sent_idx-1
                
                if self.config.use_topic==1:
                    if sent_idx==0:
                        Ts = self.initTs
                        
                    else:
                        S_i = S_sents[prev_sent_idx]

                        
                        
                        pw_zd = psi[:, input_words[doc_idx, prev_sent_idx, :]].T # words, T
                        numerator = (theta[doc_idx] * pw_zd) # theta: 1,T; pw_zd: words, T
                        denominator = torch.sum(numerator, -1).unsqueeze(-1)
                        pz_wd = numerator / denominator

                        topic_emb_w = torch.matmul(pz_wd, (self.topic_embeddings)) # max_words_doc,T * T, 768

                        topic_emb_w = topic_emb_w*input_w_mask[doc_idx][prev_sent_idx].unsqueeze(-1)

                        topic_feature_prev = topic_emb_w.sum(0)#/(input_w_mask[doc_idx][prev_sent_idx].sum())
                        
                        topic_feature_sents = topic_feature_sents.clone()
                        topic_feature_sents[prev_sent_idx, :] = topic_feature_prev
                        
                        ## get Ts[0->sent_index-1] 
                        all_prev_ts = topic_feature_sents[:prev_sent_idx+1, :].view(-1, 768)
                        score_ = torch.mm(all_prev_ts, self.weight_ts.T) # # sent_index-1, 768, 768,1
                        # alpha_ = torch.softmax(score_, -1)
                        alpha_ = torch.softmax(score_, 0)
                        
                        Ts = torch.mm(alpha_.T, all_prev_ts)##all_prev_ts.sum(0)#

                    S_sents.append(torch.cat((h_sents[doc_idx][sent_idx].reshape(1,768), Ts.reshape(1, 768)), -1)) # (1,768*2)
                else:
                    S_sents.append(((h_sents[doc_idx][sent_idx].reshape(1,768))))

            S_sents = torch.stack(S_sents).squeeze() # sents, 768*2
            
            gru_output = self.fc_sep2(torch.relu((self.fc_sep1(S_sents))))
            
#             gru_output = self.fc_sep3(torch.relu(self.fc_sep2(torch.relu(self.fc_sep1(S_sents)))))
#             gru_output = self.fc_sep3(S_sents)
            out_rnn_docs[doc_idx,:total_sents,:] = gru_output.squeeze(0)
        
        logits = self.drop_fc(self.fc_logits(out_rnn_docs))
        ## Draw L
        masked_sent_prop = F.softmax(logits, -1)*input_sent_mask.unsqueeze(2).repeat(1,1,self.config.num_intents)
        if train:
            sent_groundtruth_batch_onehot = torch.nn.functional.one_hot((sent_groundtruth_batch*input_sent_mask).long(), num_classes=self.config.num_intents)
            masked_sent_label = sent_groundtruth_batch_onehot * input_sent_mask.unsqueeze(2).repeat(1,1,self.config.num_intents) # batch, sents, I
            
            # Betalds
            sent_label_intent = masked_sent_label.unsqueeze(2).repeat(1,1,self.config.max_word_sent,1) #([14, 28, 5])->([14, 28, 73, 5]) # bs, sents, words, intent
            beta_lds = torch.matmul(sent_label_intent, torch.softmax(self.intent_proportion, dim=-1)) # bs, sents, words, V
            
            
            
            recon_topic = torch.matmul(theta.unsqueeze(1).repeat(1, self.config.max_sent_num, 1), psi) # BS, V
            recon_topic = recon_topic.unsqueeze(2).repeat(1,1,self.config.max_word_sent, 1) # BS, sents, words, V
            
            
            
            masked_input_onehot = input_one_hot.float()*input_w_mask.unsqueeze(3)

            total_ll = torch.sum((self.bernoulii_p*recon_topic + (1-self.bernoulii_p)*beta_lds+1e-20).log()* masked_input_onehot.float(), dim=-1)

            return  None, total_ll, KL_loss_MLP_arr, None, theta, masked_sent_prop, None, None
        else:
            if compute_perp:

                masked_sent_label = F.gumbel_softmax(logits, tau=0.1, hard=True)*input_sent_mask.unsqueeze(2).repeat(1,1,self.config.num_intents)
                
                sent_label_intent = masked_sent_label.unsqueeze(2).repeat(1,1,self.config.max_word_sent,1) #([14, 28, 5])->([14, 28, 73, 5]) # bs, sents, words, intent
                beta_lds = torch.matmul(sent_label_intent, torch.softmax(self.intent_proportion, dim=-1)) # bs, sents, words, V


                recon_topic = torch.matmul(theta.unsqueeze(1).repeat(1, self.config.max_sent_num, 1), psi) # BS, V
                recon_topic = recon_topic.unsqueeze(2).repeat(1,1,self.config.max_word_sent, 1) # BS, sents, words, V

                masked_input_onehot = input_one_hot.float()*input_w_mask.unsqueeze(3)
                total_ll = torch.sum((self.bernoulii_p*recon_topic + (1-self.bernoulii_p)*beta_lds+1e-20).log()* masked_input_onehot.float(), dim=-1)

                return  None, total_ll, None, None, theta, masked_sent_prop, None, None 
            else:
                return None, None, None, None, theta, masked_sent_prop, None, None
   
