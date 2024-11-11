import os
import numpy as np
import torch

# outputs: [150, 26, 7]
# labels: [150, 26]
def loss_supervised_gt(outputs, labels, num_intents):

    outputs = outputs.view(-1, num_intents).float()
#     outputs = F.log(outputs)
    outputs = (1e-12+outputs).log()
    labels = labels.view(-1).long()

    #mask out 'PAD' tokens
    mask = (labels >= 0).long()

    #the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask))

    #pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels]*mask

    #cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)

def get_docs(path, meta_path, text_path, intent_label_path):
    path = path#'./data/chem/'
    meta_path = meta_path#path+"chemical.meta"
    text_path = text_path#path+"chemical.text"
    intent_label_path = intent_label_path#path+"chemical.label"
    # Default word tokens
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    class Voc:
        def __init__(self, name):
            self.name = name
            self.trimmed = False
            self.word2index = {"[PAD]": PAD_token}
            self.word2count = {}
            self.index2word = {PAD_token: "[PAD]"}#{PAD_token: "[PAD]"}#{PAD_token: "[PAD]", SOS_token: "SOS"}#{PAD_token: "[PAD]", SOS_token: "SOS"}#, EOS_token: "EOS"}#{PAD_token: "[PAD]"}#{PAD_token: "[PAD]", SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = len(self.index2word)  # Count PAD

        def addSentence(self, sentence):
            for word in sentence.split(' '):

                self.addWord(word)

        def addWord(self, word):

            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.word2count[word] = 1
                self.index2word[self.num_words] = word
                self.num_words += 1
            else:
                self.word2count[word] += 1

        def trim_wc(self, min_count):

            keep_words = []
    #         sort_wc = dict(sorted(voc.word2count.items(), key=lambda item: item[1], reverse=True))
            sort_wc = dict(sorted(self.word2count.items(), key=lambda item: item[1], reverse=True))

            keep_words = [e for e in list(sort_wc.keys())[:min_count]]

            # Reinitialize dictionaries
            self.word2index = {}
            self.word2count = {}
            self.index2word = {}#{PAD_token: "[PAD]", SOS_token: "SOS"}#{PAD_token: "[PAD]", SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = len(self.index2word)#3 # Count default tokens

            for word in keep_words:
                self.addWord(word)

        # Remove words below a certain count threshold
        def trim(self, min_count):
            if self.trimmed:
                return
            self.trimmed = True

            keep_words = []

            for k, v in self.word2count.items():
                if v >= min_count:
                    keep_words.append(k)

            print('keep_words {} / {} = {:.4f}'.format(
                len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
            ))

            # Reinitialize dictionaries
    #         self.word2index = #{}
    #         self.word2count = {}
    #         self.index2word = self.index2#{}#{PAD_token: "[PAD]"}#{PAD_token: "[PAD]", SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = len(self.index2word)#3 # Count default tokens

            for word in keep_words:
                self.addWord(word)
    intent_groundtruth_labels = []
    with open(intent_label_path, errors='ignore') as f:
        for idx, line in enumerate(f):
            intent_groundtruth_labels.append(int(line.strip()))
    print(len(intent_groundtruth_labels))


    def get_max_doc_sen(meta_path):
        #meta_path = "data/chem/chemical.meta"
        doc_max_sen = 0
        with open(meta_path, errors='ignore') as f:
            for idx, line in enumerate(f):
                cur_len = int(line)
                if cur_len > doc_max_sen:
                    doc_max_sen = cur_len
        return doc_max_sen
    def get_max_sen_len(data_path):
        #data_path = "data/chem/chemical.text"
        sen_max_len = 0

        with open(data_path, errors='ignore') as f:
            for idx, line in enumerate(f):
                cur_len = len(line.split())
                if cur_len > sen_max_len:
                    sen_max_len = cur_len
        return sen_max_len


    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(s):
        s = s.lower().strip()#unicodeToAscii(s.lower().strip())
        #s = re.sub(r"([.!?])", r" \1", s)
        #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        #s = re.sub(r"\s+", r" ", s).strip()
        return s

    # Read query/response pairs and return a voc object
    def readVocs(datafile, corpus_name):
        print("Reading lines...")
        # Read the file and split into lines
        lines = open(datafile, encoding='utf-8', errors='ignore').read().strip().split('\n')

        # Split every line into pairs and normalize
        sen = [normalizeString(l) for l in lines]
        voc = Voc(corpus_name)
        return voc, sen


    # Using the functions defined above, return a populated voc object and pairs list
    def loadPrepareData(corpus_name, datafile, save_dir):
        print("Start preparing training data ...")
        voc, sens = readVocs(datafile, corpus_name)

        print("Counting words...")
        for sen in sens:
            voc.addSentence(sen)
    #     voc.trim_wc(5000)
        print("Counted words:", voc.num_words)
        return voc, sens


    corpus_name = "chem"
    save_dir = os.path.join("data", "save")
    voc, sens = loadPrepareData(corpus_name, text_path, save_dir)



    max_sen_len = get_max_doc_sen(meta_path) #+ 2 # total words + SOS + EOS
    max_doc_len = get_max_sen_len(text_path) #+ 2 # total sents + S + E

    sent_groundtruth = []
    docs_ = []
    docs_w = []
    docs_mask = []
    sents_mask = []
    docs_len = []
    doc_idx = 0
    total_intents_from_meta = 0
    with open(meta_path, errors='ignore') as file, open(text_path, errors='ignore') as f2, open(intent_label_path, errors='ignore') as f3:
        for idx, line in enumerate(file):
            doc = []
            senNum = int(line)
            #doc_sents[idx] = [curSenNum, curSenNum+senNum]
            #curSenNum+=senNum
            sen_idx = 0
            doc_ = np.zeros((max_sen_len, max_doc_len))

            doc_mask_ = np.zeros((max_sen_len, max_doc_len))
            doc_len = np.zeros(max_sen_len)
            doc_w = []
            sent_mask = np.zeros(max_sen_len)
            #sent_groundtruth = np.zeros(max_sen_len)
            sent_label_each_doc = -1*torch.ones(max_sen_len)
            for i in range(senNum):
                #sent_mask[i] = 1
                label_line = f3.readline().strip().lower()
                total_intents_from_meta = max(total_intents_from_meta, int(label_line))
                sent_label_each_doc[i] = int(label_line)-1
                line_data = f2.readline().strip().lower()
                line_data_split = line_data.split()
                list_valid_words = []
                list_valid_words_idx = []
                list_mask = []
                for word in line_data_split:
                    if word in voc.word2index:
                        list_valid_words_idx.append(voc.word2index[word])
                        list_valid_words.append(word)
                        list_mask.append(1)

                doc_[sen_idx, :len(list_valid_words)] = list_valid_words_idx#[voc.word2index[word] for word in line_data_split] # [SOS_token] + 
                doc_w.append(list_valid_words) # [SOS_token] + 
                doc_mask_[sen_idx, :len(list_mask)] = list_mask#[1 for word in line_data_split]

                doc_len[sen_idx] = len(line_data_split) #+ 1
                sen_idx+=1

            sent_groundtruth.append(sent_label_each_doc)
            sent_mask[:senNum] = [1 for word in range(senNum)]
            sents_mask.append(sent_mask)
            docs_.append(doc_)
            docs_w.append(doc_w)
            docs_mask.append(doc_mask_)
            docs_len.append(doc_len)
    return docs_w, docs_, docs_len, docs_mask, sents_mask, sent_groundtruth, voc, max_sen_len, max_doc_len, total_intents_from_meta