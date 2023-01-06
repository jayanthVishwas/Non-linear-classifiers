#author - Jayanth

import json
import numpy as np
from operator import methodcaller
from tqdm import tqdm,trange
import pickle

def checkOverlapping(arr):
    for i in range(len(arr)):
        x,y = arr[i]
        for j in range(i+1,len(arr)):
            a,b = arr[j]
            if (x<a and a<y and y<b) or (a<x and x<b and b<y):
                return True
    return False

def checkChildrensAndAppend(stack):

    for i in range(3):
        l = len(stack[i].childrens)
        if l <4:
            for j in range(4 - l):
                stack[i].childrens.append(None)
    return stack

def getChildrenFeatures(stack):
    features=[]
    # the words, POS tags, and arc labels of the first and second leftmost and rightmost children of the first two words on the stack
    for i in range(2):
        for j in range(2):
            if stack[i].childrens[j]==None:
                # temp=[None]*3
                temp = ["Null_Word", "Null_Pos", "Null_Arc"]
                features.extend(temp)
            else:
                temp=stack[i].childrens[j]
                features.extend([temp.word,temp.pos,temp.arc_label])

    for i in [-1,-2]:
        for j in range(2):
            if stack[i].childrens[j]==None:
                # temp=[None]*3
                temp = ["Null_Word", "Null_Pos", "Null_Arc"]
                features.extend(temp)
            else:
                temp=stack[i].childrens[j]
                features.extend([temp.word,temp.pos,temp.arc_label])

    # the words, POS tags, and arc labels of leftmost child of the leftmost child and rightmost
    # child of rightmost child of the first two words of the stack



    if stack[0].childrens[0]==None or stack[0].childrens[0].childrens[0]==None:
        # temp=[None]*3
        temp = ["Null_Word", "Null_Pos", "Null_Arc"]
        features.extend(temp)
    else:
        temp=stack[0].childrens[0].childrens[0]
        features.extend([temp.word,temp.pos,temp.arc_label])

    if stack[1].childrens[0]==None or stack[1].childrens[0].childrens[0]==None:
        # temp=[None]*3
        temp = ["Null_Word", "Null_Pos", "Null_Arc"]
        features.extend(temp)
    else:
        temp=stack[1].childrens[0].childrens[0]
        features.extend([temp.word,temp.pos,temp.arc_label])

    if stack[0].childrens[-1]==None or stack[0].childrens[-1].childrens[-1]==None:
        # temp=[None]*3
        temp = ["Null_Word", "Null_Pos", "Null_Arc"]
        features.extend(temp)
    else:
        temp=stack[0].childrens[-1].childrens[-1]
        features.extend([temp.word,temp.pos,temp.arc_label])

    if stack[1].childrens[-1]==None or stack[1].childrens[-1].childrens[-1]==None:
        # temp=[None]*3
        temp = ["Null_Word", "Null_Pos", "Null_Arc"]
        features.extend(temp)
    else:
        temp=stack[1].childrens[-1].childrens[-1]
        features.extend([temp.word,temp.pos,temp.arc_label])


    return features



class wordObj():
    def __init__(self, list_arr):
        self.id= int(list_arr[0])
        self.word=list_arr[2]
        self.pos=list_arr[4]
        self.parent_id = int(list_arr[6])
        self.arc_label = list_arr[7]
        self.childrens = []


class sentenceObj():
    def __init__(self):
        self.words=[]
        self.words_id_parId = []

    def append(self, word):
        self.words.append(word)

    def appendID(self, arr):
        self.words_id_parId.append(arr)

    def getWords(self):
        return [word.word for word in self.words]


class sentencesObj():
    def __init__(self):
        self.sentences=[]

    def append(self, sentence):
        self.sentences.append(sentence)


class ParseFile():
    def __init__(self, file):
        self.file_path=file
        self.sentences=[]
        self.word_vocab = set()
        self.pos_vocab = set()
        self.arc_vocab = set()
        self.transitions_vocab = ["SHIFT", "REDUCE", "LEFT", "RIGHT"]
        self.transitions2id = {w: k for (k, w) in enumerate(self.transitions_vocab)}
        self.word2id = {}
        self.pos2id={}
        self.arc2id={}
        self.vocab_model={}

        self.readFile()

    def readFile(self):
        f = open(self.file_path, encoding="utf8")
        sentence = sentenceObj()
        for line in tqdm(f):

            if line != '\n':
                data = line.split('\t')
                word = wordObj(data)
                sentence.append(word)
                self.word_vocab.add(word.word)
                self.pos_vocab.add(word.pos)
                self.arc_vocab.add(word.arc_label)

                sentence.appendID([ int(data[0]) , int(data[6]) ])

            if line == '\n':
                self.sentences.append(sentence)
                sentence = sentenceObj()

        self.convertVocabToId()
        self.saveVocabs()

    def getSentences(self):
        return self.sentences

    def convertVocabToId(self):
        self.word_vocab.add("Null_Word")
        self.arc_vocab.add("Null_Arc")
        self.pos_vocab.add("Null_Pos")

        self.arc_vocab.add(" ")
        self.word_vocab.add("UNK")
        self.arc_vocab.add("UNK")
        self.pos_vocab.add("UNK")


        self.word2id = {w:k for (k,w) in enumerate(self.word_vocab)}
        self.pos2id = {w: k for (k,w) in enumerate(self.pos_vocab)}
        self.arc2id = {w:k for (k,w) in enumerate(self.arc_vocab)}

    def saveVocabs(self):
        self.vocab_model={"words_id":self.word2id, "pos_id":self.pos2id, "arc_id": self.arc2id, "transition_id": self.transitions2id}
        # with open("vocabs_pikle",'wb') as f:
        #     pickle.dump(self.vocab_model,f)

    def getVocabs(self):
        return self.vocab_model


class featuresData():
    def __init__(self):
        self.features=[]
        self.arc_transition_labels=[]
        self.unique_labels=[]


    def appendFeatures(self, feature):
        self.features.extend(feature)

    def appendTransition(self, transition):
        self.arc_transition_labels.extend(transition)

    def getUniqLabels(self):
        self.unique_labels = set(self.arc_transition_labels)
        return self.unique_labels


class arcEagerShiftParser():
    def __init__(self, sentence):
        self.sentence= list(sentence.words)
        self.stack=[wordObj([0,None,'Root',None,None,None,-1,None])]
        # self.stack = []
        self.buffer= list(sentence.words)
        self.arc_transitions=[]
        self.features=[]
        self.childPar=[]

    def parse(self):
        while(self.buffer or self.stack):
            # if len(self.stack)==0:
            #     if self.buffer[0].parent_id==0:
            #         #right
            #         self.rightArcShift(self.buffer[0].arc_label)
            #     else:
            #         #shift
            #         self.shift()
            # else:
            if self.buffer and self.stack and self.buffer[0].parent_id==self.stack[-1].id:
                #right
                self.getFeatures()
                self.rightArcShift(self.buffer[0].arc_label)

            elif self.buffer and self.stack and self.stack[-1].parent_id==self.buffer[0].id:
                #left
                self.getFeatures()
                self.leftArcShift(self.stack[-1].arc_label)

            elif self.buffer:
                self.getFeatures()
                self.shift()
            else:
                #reduce
                self.getFeatures()
                self.reduce()

    def shift(self):
        self.arc_transitions.append("SHIFT")
        self.stack.append(self.buffer.pop(0))

    def reduce(self):
        self.arc_transitions.append("REDUCE")
        self.stack.pop()

    def leftArcShift(self, arc):
        self.buffer[0].childrens.append(self.stack[-1])
        self.arc_transitions.append("LEFT_"+arc)
        self.stack.pop()

    def rightArcShift(self,arc):
        self.stack[-1].childrens.append(self.buffer[0])
        self.arc_transitions.append("RIGHT_"+arc)
        self.stack.append(self.buffer.pop(0))

    def checkSizeAndAppend(self, arr):
        l = len(arr)
        # null_obj =wordObj([0, None, None, None, None, None, -1, None])
        null_obj = wordObj([0, None, "Null_Word", None, "Null_Pos", None, -1, "Null_Arc"])
        if l<3:
            for i in range(3-l):
                arr.append(null_obj)
        return arr


    def getFeatures(self):
        stack_len = len(self.stack)
        buffer_len = len(self.buffer)
        rev_stack = list(self.stack)
        if(rev_stack):
            rev_stack.pop(0)
        rev_stack.reverse()
        feature_temp=[]
        buffer_temp = list(self.buffer)

        rev_stack = self.checkSizeAndAppend(rev_stack)
        buffer_temp = self.checkSizeAndAppend(buffer_temp)

        #get words and pos of first three words on stack and buffer
        feature_temp.extend([rev_stack[0].word,rev_stack[1].word,rev_stack[2].word, rev_stack[0].pos, rev_stack[1].pos, rev_stack[2].pos ])
        feature_temp.extend([buffer_temp[0].word, buffer_temp[1].word, buffer_temp[2].word, buffer_temp[0].pos, buffer_temp[1].pos,buffer_temp[2].pos])

        rev_stack = checkChildrensAndAppend(rev_stack)

        #the words, POS tags, and arc labels of the first and second leftmost and rightmost children of the first two words on the stack
        # the words, POS tags, and arc labels of leftmost child of the leftmost child and rightmost
        # child of rightmost child of the first two words of the stack
        feature_temp.extend(getChildrenFeatures(rev_stack))

        self.features.append('\t'.join(feature_temp))


    def getTransactions(self):
        return self.arc_transitions



if __name__ == "__main__":


    parse_obj = ParseFile("work/dev.orig.conll")
    model_pikle = parse_obj.getVocabs()

    data = parse_obj.getSentences()
    features = featuresData()

    featuresFile = "features-dev.txt"

    labels = []
    projective_count=0
    non_projective_count=0
    with open(featuresFile, 'w+') as f:
        for i in trange(len(data)):
            if not checkOverlapping(data[i].words_id_parId):
                projective_count+=1
                p = arcEagerShiftParser(data[i])
                p.parse()
                for arc_trans, feature in zip(p.arc_transitions, p.features):
                    f.write(f'{arc_trans}\t{feature}\n')

                labels.extend(p.arc_transitions)
                # features.appendFeatures(p.features)
                # features.appendTransition(p.arc_transitions)
            else:
                non_projective_count+=1
    print(f"Number of projective sentences:{projective_count}")
    print(f"Number of non projective sentences:{non_projective_count}")
    model_pikle["unique"] = set(labels)
    # model_pikle['features'] = np.array(features.features)
    # model_pikle["labels"] = features.arc_transition_labels
    with open("model_pikle", 'wb') as f:
        pickle.dump(model_pikle,f)

