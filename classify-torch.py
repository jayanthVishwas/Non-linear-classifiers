import pickle
import argparse
import numpy as np
from torch.optim import Adam
import torch.nn as nn

import nn_layers
from train import *
from nn_layers import *

def convertDataToEmbeddingMatrix(tokens,word_embeds,max_len,unk_vector):
    embedding_size=len(list(word_embeds.values())[0])
    X=np.zeros((len(tokens), max_len*embedding_size), dtype='float32')
    zeros=[0]*(embedding_size)
    for i,texts in enumerate(tokens):
        texts=padOrTruncate(texts,max_len)
        temp=[]
        for text in texts:
            if text in word_embeds:
                temp =temp + list(word_embeds[text])
            elif text==0:
                temp = temp + zeros
            else:
                temp=temp+list(unk_vector["UNK"])
        X[i]=temp
    return X

def getDatafromFile(data_file):
    with open(data_file, encoding="utf8") as file:
        data = file.read().splitlines()
    # data_split = map(methodcaller("rsplit", "\t", 1), data)
    # texts = map(list, zip(*data))
    return data

def forward( X, w1, b1, w2, b2):
    A1 =sigmoid(X.dot(w1) + b1)
    Y = softmax(A1.dot(w2) + b2)

    return Y

def sigmoid( x):
    return 1 / (1 + np.exp(-x))

def softmax( x):
    expx = np.exp(x)
    return expx / expx.sum(axis=1, keepdims=True)

def getPredictedClass(ypred,classes):
    y=[]
    for i in range(len(ypred)):
        temp=classes[np.argmax(ypred[i])]
        y.append(temp)

    return y



def classify(args):
    model_file = args.m
    with open(model_file, "rb") as file:
        model_dict = pickle.load(file)


    model=model_dict['model']
    max_len=model_dict['max_len']
    classes = model_dict['classes']
    emb_file=model_dict['emb_file']
    unk_vector=model_dict['unk_vector']

    word_embeddings = getEmbeddings(emb_file)

    input_file=args.i
    texts = getDatafromFile(args.i)
    tokenized_text = [tokenize(text) for text in texts]

    # if "odia" in input_file:
    #     unk_vector=getEmbeddings("unk-odia.vec")
    #
    # else:
    #     unk_vector=getEmbeddings("unk.vec")
    #     word_embeddings = getEmbeddings("glove.6B.50d.txt")

    X = convertDataToEmbeddingMatrix(tokenized_text, word_embeddings, max_len, unk_vector)
    X=torch.FloatTensor(X)
    ypred=model(X)
    ypred= getPredictedClass(ypred.detach().numpy(),classes)

    with open(args.o,'w',encoding="utf8") as file:
        for y in ypred:
            file.write(y)
            file.write('\n')


    with open("datasets/questions-label-test.txt",encoding="utf8") as file:
        data = file.readlines()

    label=[]
    for i in range(len(data)):
        label.append(data[i].strip())


    label=np.array(label)
    ypred=np.array(ypred)
    acc = np.mean(ypred==label)
    print(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net inference arguments.')

    parser.add_argument('-m', type=str, help='trained model file')
    parser.add_argument('-i', type=str, help='test file to be read')
    parser.add_argument('-o', type=str, help='output file')

    args = parser.parse_args()

    classify(args)
