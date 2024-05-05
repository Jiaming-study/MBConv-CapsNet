"""
Created on Sun Jan  5 21:11:08 2021

@author: Mr.JXD
"""
import numpy as np
import h5py
import re
import operator

import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
from collections import defaultdict
from nltk import word_tokenize

from nltk.corpus import stopwords
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
cachedStopWords = stopwords.words("english")

#print(cachedStopWords)

def load_word_vector(fname, vocab):
    model = {}
    with open(fname) as fin:
        for line_no, line in enumerate(fin):
            try:
                parts = line.strip().split(' ')
                word, weights = parts[0], parts[1:]
                if word in vocab:                     
                    model[word] = np.array(weights,dtype=np.float32)
            except:
                pass
    return model

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for _ in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word).decode('utf-8')
                    break
                if ch != b'\n':
                    word.append(ch)
            if word in vocab: 
                word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs
#句子变成词
def line_to_words(line):
    words = map(lambda word: word.lower(), word_tokenize(line))
    #print("words",words)    
    tokens = words
    p = re.compile('[a-zA-Z]+')
    return list(filter(lambda token: p.match(token) and len(token) >= 3, tokens))        

def get_vocab(dataset):
    max_sent_len = 0
    word_to_idx = {}
    idx = 1
    for line in dataset:    
        words = line_to_words(line)
        #print("words",words)
        max_sent_len = max(max_sent_len, len(words))
        for word in words:
            if not word in word_to_idx:
                word_to_idx[word] = idx
                idx += 1
    return max_sent_len, word_to_idx

def load_txt(path_name):
    with open(path_name, 'r', encoding='ISO-8859-1') as f:
        sentences = f.readlines()
    data, labels = [], []
    for sentence in sentences:
        if sentence is None or len(sentence) <= 1:
            continue
        label, _, text = sentence.partition(' ')
        data.append(text.strip())   
        labels.append(label.strip())  
    return data, labels

def load_data(train_path, test_path, padding=0, sent_len=300, w2i=None):
    train_data, train_labels = load_txt(train_path)
    test_data, test_labels = load_txt(test_path)
#     popular_topics = set(['World','Sprorts','Business','Sci/Tec'])
    
    dataset = train_data + test_data
    max_sent_len, word_to_idx = get_vocab(dataset)
    print('max_sent_len:',max_sent_len)
    if sent_len > 0:
        max_sent_len = sent_len      
    if w2i is not None:
        word_to_idx = w2i    

    train, train_label = [], []
    test, test_label = [], []
    print('train_data:',len(train_data))     
    print('test_data:',len(test_data))   
    for i, text in enumerate(train_data):
        words = line_to_words(text)
        length=str(len(words))
        y = train_labels[i]
        sent = [word_to_idx[word] for word in words if word in word_to_idx]
        if len(sent) > max_sent_len:
            sent = sent[:max_sent_len]
        else:    
            sent.extend([0] * (max_sent_len + padding - len(sent)))
        train.append(sent)
        train_label.append(y)
    single_label = ['-1'] + list(set(train_label))
    print('label:',single_label)
    num_classes = len(single_label)
    print(num_classes)
    for i, l in enumerate(train_label):
        train_label[i] = single_label.index(l) 
    for i, text in enumerate(test_data):
        words = line_to_words(text)
        y = test_labels[i]
        sent = [word_to_idx[word] for word in words if word in word_to_idx]
        if len(sent) > max_sent_len:
            sent = sent[:max_sent_len]
        else:    
            sent.extend([0] * (max_sent_len + padding - len(sent)))
        test.append(sent)
        one_hot_y = np.zeros([num_classes],dtype=np.int32)
        for yi in y:
            one_hot_y[single_label.index(yi)]=1
        test_label.append(one_hot_y)
    
    return single_label, word_to_idx, np.array(train), np.array(train_label), np.array(test), np.array(test_label)

train_path = 'data/agnews.train.all'  # 训练集路径
test_path = 'data/agnews.test.all'  # 测试集路径

single_label, word_to_idx, train, train_label, test, test_label = load_data(train_path, test_path, padding=0, sent_len=133, w2i=None)

#print(test_label)
print(train[0])
print(single_label)
print ('train size:', train.shape)
print ('test size:', test.shape)
print ('train_lable size:', train_label.shape)
print ('test_lable size:', test_label.shape)


dataset = 'agnews'

with open(dataset + '_word_mapping.txt', 'w+') as embeddings_f:
    embeddings_f.write("*PADDING* 0\n")
    for word, idx in sorted(word_to_idx.items(), key=operator.itemgetter(1)):
      embeddings_f.write("%s %d\n" % (word, idx))

w2v_path = 'GoogleNews-vectors-negative300.bin'
#glove_path = 'glove.6B/glove.6B.300d.txt'

w2v = load_bin_vec(w2v_path, word_to_idx)
#print(w2v)
vector_length = len(next(iter(w2v.values()))) if w2v else 0 
V = len(word_to_idx) + 1
print ('Vocab size:', V) 

def compute_embed(V, w2v, vector_length):  
    np.random.seed(1)  
    embed = np.random.uniform(-0.25, 0.25, (V, vector_length))   
    # 用 w2v 中的词向量填充嵌入矩阵  
    for word, vec in w2v.items():  
        if word in word_to_idx and word_to_idx[word] != 0:  # 跳过填充词  
            embed[word_to_idx[word]] = vec  
      
    return embed  
  
# 计算嵌入矩阵  
embed_w2v = compute_embed(V, w2v, vector_length) 
#embed_glove = compute_embed(V, glove)

print ('train size:', train.shape)
print ('test size:', test.shape)

filename = dataset + '.hdf5'
with h5py.File(filename, "w") as f:
    f["w2v"] = np.array(embed_w2v)
    f['train'] = train
    f['train_label'] = train_label
    f['test'] = test
    f['test_label'] = test_label

