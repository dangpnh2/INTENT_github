#!/usr/bin/env python
# coding: utf-8


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from types import SimpleNamespace
import matplotlib.pyplot as plt
from io import open
import torch
from collections import Counter
import numpy as np
from typing import Tuple, Dict, Union, List
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from load_data_chem_format import get_docs
from model import SentSeq, VAEOneTheta

# In[2]:


import transformers


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--hidden_size", type=int, default=100)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--bs", type=int, default=16)
parser.add_argument("--print_every", type=int, default=3)
parser.add_argument("--num_topics", type=int, default=5)
parser.add_argument("--lr1", type=float, default=0.001)
parser.add_argument("--lr2", type=float, default=5e-5)
parser.add_argument("--wd", type=float, default=0.001)
parser.add_argument("--save_model", type=int, default=0)
parser.add_argument('--device',type=str,default='cuda')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_topic", type=int, default=1)

parser.add_argument('--dataset_name',type=str,default='chemical')
parser.add_argument('--name1',type=str,default='1')
parser.add_argument('--name2',type=str,default='2')
args = parser.parse_args()

# In[3]:


print(args)
import time
torch.manual_seed(int(time.time()))



data_name = args.dataset_name#"chemical"
path = './data/{}/'.format(data_name)
meta_path = path+"{}.meta".format(data_name)
text_path = path+"{}.text".format(data_name)
intent_label_path = path+"{}.label".format(data_name)

device = args.device #"cuda"
docs_w, docs_, docs_len, docs_mask, sents_mask, sent_groundtruth, voc, max_sen_len, max_doc_len, total_intents_from_meta = get_docs(path, meta_path, text_path, intent_label_path)


sent_groundtruth = torch.stack(sent_groundtruth, 0)
tensor_X =  torch.LongTensor(np.array(docs_))
tensor_docs_len = torch.tensor(np.array(docs_len))
tensor_word_mask = torch.tensor(np.array(docs_mask)).float()
tensor_sent_mask = torch.tensor(np.array(sents_mask)).float()



count_first_label = {}
for i in sent_groundtruth:
    if i[0].item() not in count_first_label:
        count_first_label[i[0].item()] = 0
    
    count_first_label[i[0].item()] += 1



class MLPConfig(object):
    use_topic = args.use_topic
    num_input = voc.num_words#tensor_X_MLP.shape[1]#voc.num_words
    embed_size = 768
    vocab_size = voc.num_words
    max_sent_num = max_sen_len
    max_word_sent = max_doc_len
    num_topics = args.num_topics
    num_intents = total_intents_from_meta
    device = device
    drop_rate = 0.0
    hidden_size = args.hidden_size
    h1 = args.hidden_size
    h2 = args.hidden_size
    encoder_n_layers = 1
    learning_rate = args.lr1
    learning_rate_sent = args.lr2
    batch_size = args.bs
    bert_model_name = "allenai/scibert_scivocab_uncased"
#     bert_model_name = "dmis-lab/biobert-v1.1"
    bert_model = None
    tokenizer = None

mlpconfig = MLPConfig()



all_len  = []
for doc in tensor_X:
    all_len.append(torch.count_nonzero(doc))
print(np.array(all_len).max())


# In[21]:


## PROCESSING BERT (Convert input data to Bert input)
tokenizer = BertTokenizer.from_pretrained(mlpconfig.bert_model_name)
mlpconfig.tokenizer = tokenizer
max_len_words_per_doc = 512


bert_input = torch.zeros((tensor_X.shape[0], max_len_words_per_doc))
bert_input_mask = torch.zeros((tensor_X.shape[0], max_len_words_per_doc))

labels_bert_format = []
for doc_idx, doc in enumerate(tensor_X):
    doc_words = []
    doc_labels_bert = []
    for sent_idx, sent in enumerate(doc):
        sent_words =  [voc.index2word[w] for w in sent.detach().cpu().numpy() if w!=0] 
        
        if len(sent_words)>0:
            doc_words = doc_words+sent_words+ ['[SEP]'] # sent1 + [SEP] + sen2 + [SEP] +...+ [SEP]
            doc_labels_bert.append(sent_groundtruth[doc_idx][sent_idx].item())
    doc_words = ' '.join(e for e in doc_words[:-1])
    embeded_sent = tokenizer(doc_words, return_tensors='pt', max_length=max_len_words_per_doc, padding='max_length', truncation=True)    
    bert_input[doc_idx,:] = embeded_sent['input_ids']
    bert_input_mask[doc_idx,:] = embeded_sent['attention_mask']
    labels_bert_format.append(doc_labels_bert)
labels_bert_format = np.array(labels_bert_format, dtype=object)


## MODELS PARAMS

learning_rate = mlpconfig.learning_rate
num_intents = mlpconfig.num_intents
hidden_size = mlpconfig.hidden_size
embedding_size = mlpconfig.embed_size
encoder_n_layers = mlpconfig.encoder_n_layers
dropout = mlpconfig.drop_rate
batch_size = mlpconfig.batch_size


sent_seq_model = SentSeq(mlpconfig, hidden_size, num_intents, mlpconfig.max_word_sent, mlpconfig.embed_size,  1, dropout, mlpconfig.device).to(device)
sent_seq_model.train()

## Get embedding of each word in current vocab from scibert vocab
word_emb_bert = sent_seq_model.bert.embeddings.word_embeddings.weight # vocab_size_scibert x 768
word_index_in_bert = sent_seq_model.config.tokenizer.convert_tokens_to_ids(list(voc.word2index.keys())) # current_voc_size (convert current voc to scibert vocab index; [PAD]=>0; paper->1203)

embedding_matrix=torch.zeros((len(word_index_in_bert), word_emb_bert.shape[1])).to(device) #current_voc_size x 768
for index, bert_token in enumerate(word_index_in_bert):
    embedding_matrix[index] = word_emb_bert[bert_token]
    
    
tensor_pretrained_emb = torch.tensor(embedding_matrix).float().to(device)

mlps = VAEOneTheta(mlpconfig, tensor_pretrained_emb).to(device)
mlps = mlps.float()
mlps.train()




opt = optim.Adam([
                {'params': sent_seq_model.parameters(), 'lr': mlpconfig.learning_rate_sent, 'weight_decay': args.wd}, #, 
            {'params': mlps.parameters(), 'lr': mlpconfig.learning_rate, 'weight_decay': args.wd}
            ])



## SPLIT TRAIN/TEST
all_indices = torch.arange(tensor_X.size(0)).split(mlpconfig.batch_size)
train_valid_idx, test_idx = train_test_split(list(range(len(all_indices))), shuffle=True, test_size=0.2, random_state=42)
train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.25, random_state=42)



all_indices_test = []
all_indices_train = []
all_indices_valid = []
for i in test_idx:
    all_indices_test.append(all_indices[i])
    
for i in train_idx:
    all_indices_train.append(all_indices[i])
    
for i in valid_idx:
    all_indices_valid.append(all_indices[i])
    


from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score

##==========NPMI============
from nltk.stem import PorterStemmer

stem = PorterStemmer()
def create_pairs(list_words, pairs):
    for i in list_words:
        for j in list_words:
            if i!=j:
                if (i+" "+j not in pairs) and (j+" "+i not in pairs):
                    pairs.update({i+" "+j:0})
    return pairs
def cal_npmi(pairs, score_dict):
    npmi = []
    for pair in pairs:
        if pair in score_dict:
            npmi.append(score_dict[pair])
        else:
            npmi.append(0)
    return npmi
# with open("./data/npmi_cached_pairs_wiki.txt", "r") as fd:
#     pair_score = fd.read().splitlines()
# pair_score_dict = {i.split('\t')[0]: i.split('\t')[1] for i in pair_score}
##==========NPMI============

def get_topwords(beta, id_vocab, nwords=10):
    topic_indx = 0
    topwords_topic = []
    npmi = []
    for i in range(len(beta)):
        pairs = {}
        topwords = " ".join([id_vocab[j] for j in beta[i].argsort()[:-nwords - 1:-1]])
        topwords_topic.append( str(topic_indx)+": "+ topwords)
        topic_indx+=1
        
        ## NPMI
        topic_split = [stem.stem(w) for w in topwords.split()]
        pairs = create_pairs(topic_split, pairs)
        # npmi_topic = np.array(cal_npmi(pairs, pair_score_dict), dtype=np.float64)
        # npmi.append(npmi_topic.mean())
    score = 0#np.mean(npmi)
    return topwords_topic, score

def test(test_seq_model, test_mlps, to_test_indices, print_top_words=True, print_report=False):
    test_seq_model.eval()
    test_mlps.eval()
    with torch.no_grad():
        sent_ground_truth_all = []
        sent_predicted_all = []
        sent_mask_all = []
        for batch_ndx in to_test_indices: ##242*0.4*100/3600
            if len(batch_ndx)!=batch_size:
                continue
            sent_groundtruth_batch = sent_groundtruth[batch_ndx].to(device)
            input_w = tensor_X[batch_ndx].to(device) # BxsenxT
#             input_w_onehot = tensor_onehot_X[batch_ndx].to(device)
            input_w_mask = tensor_word_mask[batch_ndx].to(device) # 14x28x73
            input_sent_mask = tensor_sent_mask[batch_ndx].to(device) # 14x28
            
            
            ## Bert
            input_bert = bert_input[batch_ndx].to(device)
            input_bert_attention = bert_input_mask[batch_ndx].to(device)
#             sent_gt_bert = labels_bert_format[batch_ndx].to(device)
            
            hsents = test_seq_model(input_bert.long(), input_bert_attention)
            KL_gumbel_b, loss_ll, KL_MLP, _, theta_list, masked_sent_prop, _, _ = mlps(input_w, input_bert.long(), input_w_mask, input_sent_mask, hsents, None, train=False)
            
#             masked_sent_prop = F.softmax(hsents, -1)*input_sent_mask.unsqueeze(2).repeat(1,1,mlpconfig.num_intents)
    
            predicted_label = torch.argmax(masked_sent_prop, -1)
            
            
            sent_ground_truth_all.extend(sent_groundtruth_batch.view(-1))
            sent_predicted_all.extend(predicted_label.view(-1))
            sent_mask_all.extend(input_sent_mask.view(-1))
    gr = []
    pred = []
    for idx, msk in enumerate(sent_mask_all):
        if msk == 1:
            gr.append(sent_ground_truth_all[idx].cpu().data.numpy())
            pred.append(sent_predicted_all[idx].cpu().data.numpy())
    f1 = f1_score(gr, pred, average='micro')
    ari = adjusted_rand_score(gr, pred)
#     label_dict = {0: 'OBJECTIVE', 1: 'RESULTS', 2: 'BACKGROUND', 3: 'CONCLUSIONS', 4: 'METHODS'}
    topic_ = (torch.matmul(test_mlps.topic_embeddings, test_mlps.bow_embeddings.weight.T)).detach().cpu().numpy()#test_mlps.psi_bn
    topic_topwords, npmi_score = get_topwords(topic_, voc.index2word, 10)
    if print_report:
        print(classification_report(gr, pred))
    
    if print_top_words:
        intent_words = (test_mlps.intent_proportion).detach().cpu().numpy()
        intent_topwords, _ = get_topwords(intent_words, voc.index2word, 10)
        print(intent_topwords)
        print('topics:')
        print(topic_topwords, ';npmi= ', npmi_score)
    return f1, ari, npmi_score


    
def test_perp(test_seq_model, test_mlps, to_test_indices, print_top_words=True, print_report=False):
    test_seq_model.eval()
    test_mlps.eval()
    perp_arr = []
    sum_likelihood = 0
    sum_tokens = 0
    with torch.no_grad():
        sent_ground_truth_all = []
        sent_predicted_all = []
        sent_mask_all = []
        for batch_ndx in to_test_indices: ##242*0.4*100/3600
            if len(batch_ndx)!=batch_size:
                continue
            sent_groundtruth_batch = sent_groundtruth[batch_ndx].to(device)
            input_w = tensor_X[batch_ndx].to(device) # BxsenxT
#             input_w_onehot = tensor_onehot_X[batch_ndx].to(device)
            input_w_mask = tensor_word_mask[batch_ndx].to(device) # 14x28x73
            input_sent_mask = tensor_sent_mask[batch_ndx].to(device) # 14x28
            
            
            ## Bert
            input_bert = bert_input[batch_ndx].to(device)
            input_bert_attention = bert_input_mask[batch_ndx].to(device)
#             sent_gt_bert = labels_bert_format[batch_ndx].to(device)
            
            hsents = test_seq_model(input_bert.long(), input_bert_attention)
            KL_gumbel_b, loss_ll, KL_MLP, _, theta_list, masked_sent_prop, _, _ = mlps(input_w, input_bert.long(), input_w_mask, input_sent_mask, hsents, sent_groundtruth_batch, train=False, compute_perp=True)

#             print(loss_ll.shape)
#             perp_arr.append((-loss_ll.sum(-1).sum(-1)/input_w.count_nonzero(1,2)).exp().mean().item())
            sum_likelihood+=-loss_ll.sum().item()
            sum_tokens+=input_w.count_nonzero().item()
    return np.exp(sum_likelihood/sum_tokens)

def loss_fn(outputs, labels):

    outputs = outputs.view(-1, mlpconfig.num_intents).float()
#     outputs = F.log(outputs)
    outputs = (1e-20+outputs).log()
    labels = labels.view(-1).long()

    #mask out 'PAD' tokens
    mask = (labels >= 0).long()

    #the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask))

    #pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels]*mask

    #cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)

criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
import time
start = time.time()
f1_valid_all = []
f1_test_all = []

best_f1_valid = 0
best_f1_valid_epoch = 0
best_perp = 100000
best_npmi = 0
save_path = "./saved_models/"+args.dataset_name+"_"+str(args.hidden_size)+"_"+str(args.name1)+"_"+str(args.name2)+"_"+str(args.num_topics)
def save_model(m1, m2, s_path):
    torch.save(m1.state_dict(), s_path+"seq.pt")
    torch.save(m2.state_dict(), s_path+"mlps.pt")
    
    
for epoch in range(args.epochs):
    loss_epoch = 0
    loss_label = 0
    mlps.train()
    sent_seq_model.train()
    for batch_ndx in all_indices_train: ##242*0.4*100/3600
        
        if len(batch_ndx)!=batch_size:
            continue
        sent_groundtruth_batch = sent_groundtruth[batch_ndx].to(device)
        input_w = tensor_X[batch_ndx].to(device) # BxsenxT
#         input_w_onehot = tensor_onehot_X[batch_ndx].to(device)
        input_w_mask = tensor_word_mask[batch_ndx].to(device) # 14x28x73
        input_sent_mask = tensor_sent_mask[batch_ndx].to(device) # 14x28
        
        
        ## Bert
        input_bert = bert_input[batch_ndx].to(device)
        input_bert_attention = bert_input_mask[batch_ndx].to(device)
        sent_gt_bert = torch.tensor(np.concatenate(labels_bert_format[batch_ndx]).ravel().tolist()).to(device).long()
        # TRAIN
        loss = 0
        print_losses = []
        n_totals = 0
        
        hsents = sent_seq_model(input_bert.long(), input_bert_attention) # 16, 31,768 - h for each sents

        
        KL_gumbel_b, loss_ll, KL_MLP, KL_z_docs, theta_list, masked_sent_prop, b_list, z_list = mlps(input_w, input_bert.long(), input_w_mask, input_sent_mask, hsents, sent_groundtruth_batch)

        
        
        loss_groundtruth = loss_fn(masked_sent_prop, sent_groundtruth_batch)
        
        final_loss = loss_groundtruth + ( KL_MLP.sum()) - loss_ll.sum()  #masked_KL_b.sum() +
        
        opt.zero_grad()
        final_loss.backward()             # backprop
        opt.step()
    
        loss_epoch += final_loss.item() # add loss to loss_epoch
        loss_label +=loss_groundtruth.item()
        
    
    f1_valid, _, npmi_valid = test(sent_seq_model, mlps, all_indices_valid,  print_top_words=False)
    f1_test, _, npmi_test = test(sent_seq_model, mlps, all_indices_test,  print_top_words=False, print_report=False)
    perp_epoch = test_perp(sent_seq_model, mlps, all_indices_test,  print_top_words=False, print_report=False)
    
    f1_valid_all.append(f1_valid)
    f1_test_all.append(f1_test)
    
    if f1_valid > best_f1_valid:
        best_perp = perp_epoch
        best_npmi = npmi_test
        
        best_f1_valid_epoch = epoch
        best_f1_valid = f1_valid
        
        print('new best PERP:', best_perp)
    if epoch==0:
        print(time.time()-start)
    if epoch % args.print_every==0:
        print("===========================")
        print('epoch {}, loss={}, loss_l={}'.format(epoch, loss_epoch, loss_label))
        
        print('best test f1:', f1_test_all[np.argmax(np.array(f1_valid_all))], '; at: ', best_f1_valid_epoch)
        print('perp=', best_perp)
        print('npmi= ', best_npmi)
        print('max f1 test: ', np.max(f1_test_all))
        
    if epoch % args.print_every == 0:
        
        print("train set: ")
        f1_train, _, _ = test(sent_seq_model, mlps, all_indices_train)
        
        print('f1train, f1 valid, f1 test, perp: ', f1_train, f1_valid, f1_test, perp_epoch)

print('PERP:', test_perp(sent_seq_model, mlps, all_indices_test,  print_top_words=False, print_report=False))
if args.save_model==1:
    save_model(sent_seq_model, mlps, save_path)


