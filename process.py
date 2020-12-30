import json
import csv
import jieba

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
import os
import pickle

import pandas as pd
import csv


def get_data(data_path, labels_path):
    data = []
    labels = set()
    for line in open(data_path, 'r', encoding='utf-8-sig'):
        line = json.loads(line)
        text = line['title'] + line['contents']  # 可以考虑不加contents
        label = line['classify_label']
        labels.add(label)
        data.append((text, label))
    labels = list(labels)
    labels = {i: label for i, label in enumerate(labels)}

    with open(labels_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for item in labels.items():
            s = str(item[0]) + ' ' + str(item[1])
            writer.writerow([s])
    return data, labels


data_path = '/content/drive/MyDrive/LSTM_P1/thucnews.json'
labels_path = '/content/drive/MyDrive/LSTM_P1/labels.txt'
data, labels = get_data(data_path, labels_path)
print(len(data), data[0])
print(labels)


def segment(sentence, cut_type='word', pos=False):
    if pos:
        if cut_type == 'word':
            word_pos_seq = posseg.lcut(sentence)
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)


def read_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            lines.append(line.strip('\n'))
    return lines


def save_data(data, train_data_path, test_data_path, vocab_path, path_stopwords):
    train_len, test_len = int(len(data) * 0.8), int(len(data) * 0.2)
    stopwords = read_file(path_stopwords)
    vocabs = {}
    with open(train_data_path, 'w', encoding='utf-8-sig') as f1:
        with open(test_data_path, 'w', encoding='utf-8-sig') as f2:
            writer1 = csv.writer(f1)
            writer2 = csv.writer(f2)
            counter = 0
            for line in data:
                text = line[0]
                label = line[1]
                seg_words = segment(text, cut_type='word', pos=False)
                seg_words = [word for word in seg_words if word not in stopwords]
                for word in seg_words[:128]:  # 文本太长了，处理起来太慢，先做个截断减少计算量
                    if word in vocabs:
                        vocabs[word] += 1
                    else:
                        vocabs[word] = 1
                seg_words = ' '.join(seg_words)
                if counter <= train_len:
                    writer1.writerow([seg_words, label])
                else:
                    writer2.writerow([seg_words, label])
                counter += 1

    vocabs = sorted(vocabs.items(), key=lambda d: d[1], reverse=True)
    with open(vocab_path, 'w', newline='') as f:
        writer = csv.writer(f)
        i = 0
        for item in vocabs:
            s = str(i) + ' ' + str(item[0])
            writer.writerow([s])
            i += 1
        print('the length of vocabs is : ', i)



save_data(data = data,
          train_data_path='/content/drive/MyDrive/LSTM_P1/train_data.csv',
          test_data_path='/content/drive/MyDrive/LSTM_P1/test_data.csv',
          vocab_path='/content/drive/MyDrive/LSTM_P1/vocabs.txt',
          path_stopwords='/content/drive/MyDrive/LSTM_P1/stopwords.txt')


train_data = pd.read_csv('/content/drive/MyDrive/LSTM_P1/train_data.csv',names=['text','label'])
test_data = pd.read_csv('/content/drive/MyDrive/LSTM_P1/test_data.csv',names=['text','label'])
train_union_test=pd.concat([train_data,test_data],axis=0,ignore_index=True)
with open('/content/drive/MyDrive/LSTM_P1/train_union_test.txt','w',newline='') as f:
    writer=csv.writer(f)
    for row in train_union_test.iterrows():
        s=row[1]['text']
        s=str(s)
        writer.writerow([s])


def dump_pkl(vocab, pkl_path, overwrite=True):
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("save %s ok." % pkl_path)


def build_w2v(train_union_test_path, path_words_vectors, w2v_model_path='w2v.model', min_count=5):
    w2v = Word2Vec(sentences=LineSentence(train_union_test_path), size=128, window=5, min_count=min_count, iter=5)
    w2v.save(w2v_model_path)

    model = Word2Vec.load(w2v_model_path)
    model = KeyedVectors.load(w2v_model_path)

    words_vectors = {}
    for word in model.wv.vocab:
        words_vectors[word] = model[word]
    print(len(words_vectors))
    dump_pkl(words_vectors, path_words_vectors, overwrite=True)

build_w2v(train_union_test_path='/content/drive/MyDrive/LSTM_P1/train_union_test.txt',
          path_words_vectors='/content/drive/MyDrive/LSTM_P1/words_vectors.txt')