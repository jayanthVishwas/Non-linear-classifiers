from tqdm import tqdm, trange
import torch
from preparedata import *
from model import ArcEagerParser
from copy import deepcopy
import argparse


def getWordPosArc(x, word_dict, pos_dict, arc_dict):
    word_ind = [0, 1, 2, 6, 7, 8, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]
    pos_ind = [3, 4, 5, 9, 10, 11, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46]
    arc_ind = [14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47]

    words = list(map(x.__getitem__, word_ind))
    pos = list(map(x.__getitem__, pos_ind))
    arc = list(map(x.__getitem__, arc_ind))

    words = word2index(words, word_dict)
    pos = word2index(pos, pos_dict)
    arc = word2index(arc, arc_dict)

    return words + pos + arc


def word2index(x, dict):
    res = []
    for word in x:
        res.append(dict.get(word, dict["UNK"]))
    return res


class Word():
    def __init__(self, data):
        self.id = data[0]
        self.word = data[1]
        self.lemma_word = data[2]
        self.coarsePos = data[3]
        self.finePos = data[4]
        self.morph = data[5]
        self.parId = 0
        self.arc = "root"
        # self.parId = data[6]
        # self.arc = data[7]
        self.other1 = data[8]
        self.other2 = data[9]
        self.childrens = []




def checkChildrensAndAppend(stack):
    for i in range(3):
        l = len(stack[i].childrens)
        if l < 4:
            for j in range(4 - l):
                stack[i].childrens.append(None)
    return stack


def getChildrenFeatures(stack):
    features = []
    # the words, POS tags, and arc labels of the first and second leftmost and rightmost children of the first two words on the stack
    for i in range(2):
        for j in range(2):
            if stack[i].childrens[j] == None:
                # temp=[None]*3
                temp = ["Null_Word", "Null_Pos", "Null_Arc"]
                features.extend(temp)
            else:
                temp = stack[i].childrens[j]
                features.extend([temp.lemma_word, temp.finePos, temp.arc])

    for i in [-1, -2]:
        for j in range(2):
            if stack[i].childrens[j] == None:
                # temp=[None]*3
                temp = ["Null_Word", "Null_Pos", "Null_Arc"]
                features.extend(temp)
            else:
                temp = stack[i].childrens[j]
                features.extend([temp.lemma_word, temp.finePos, temp.arc])

    # the words, POS tags, and arc labels of leftmost child of the leftmost child and rightmost
    # child of rightmost child of the first two words of the stack

    if stack[0].childrens[0] == None or stack[0].childrens[0].childrens[0] == None:
        # temp=[None]*3
        temp = ["Null_Word", "Null_Pos", "Null_Arc"]
        features.extend(temp)
    else:
        temp = stack[0].childrens[0].childrens[0]
        features.extend([temp.lemma_word, temp.finePos, temp.arc])

    if stack[1].childrens[0] == None or stack[1].childrens[0].childrens[0] == None:
        # temp=[None]*3
        temp = ["Null_Word", "Null_Pos", "Null_Arc"]
        features.extend(temp)
    else:
        temp = stack[1].childrens[0].childrens[0]
        features.extend([temp.lemma_word, temp.finePos, temp.arc])

    if stack[0].childrens[-1] == None or stack[0].childrens[-1].childrens[-1] == None:
        # temp=[None]*3
        temp = ["Null_Word", "Null_Pos", "Null_Arc"]
        features.extend(temp)
    else:
        temp = stack[0].childrens[-1].childrens[-1]
        features.extend([temp.lemma_word, temp.finePos, temp.arc])

    if stack[1].childrens[-1] == None or stack[1].childrens[-1].childrens[-1] == None:
        # temp=[None]*3
        temp = ["Null_Word", "Null_Pos", "Null_Arc"]
        features.extend(temp)
    else:
        temp = stack[1].childrens[-1].childrens[-1]
        features.extend([temp.lemma_word, temp.finePos, temp.arc])

    return features


class ParseFile():
    def __init__(self, file):
        self.sentence = []
        self.sentences = []
        self.file = file

    def readFile(self):
        with open(self.file, 'r') as f:
            sentence = []
            for line in tqdm(f):

                if line != '\n':
                    data = line.split('\t')
                    obj = Word(data)

                    sentence.append(obj)

                if line == '\n':
                    self.sentences.append(sentence)
                    sentence = []

    def getSentences(self):
        return self.sentences


def getStringData(word):
    # return "\t".join(
    #     [str(word.id), str(word.word), str(word.lemma_word), str(word.coarsePos), str(word.finePos), str(word.morph),
    #      str(word.parId), str(word.arc), str(word.other1), str(word.other2)])

    return "\t".join(
        [str(word['id']), str(word['word']), str(word['lemma_word']), str(word['coarsePos']), str(word['finePos']), str(word['morph']),
         str(word['parId']), str(word['arc']), str(word['other1']), str(word['other2'])])


class WriteFile():
    def __init__(self, sentences, out_file):
        self.sentences = sentences
        self.out_file = out_file

    def write(self):
        with open(self.out_file, "w") as f:
            for sentence in tqdm(self.sentences):
                for word in sentence:
                    str_data = getStringData(word)
                    f.write(str_data)
                f.write('\n')


class arcEagerShiftParser():
    def __init__(self, sentence,sentence2, words_dict, pos_dict, arc_dict, labels,trans_dict, model):
        self.sentence = deepcopy(sentence)

        self.stack = [Word([0, None, None, None, 'Root', None, -1, None, None, None])]
        # self.stack = []
        self.buffer = deepcopy(sentence2)
        self.arc_transitions = []
        self.features = []
        self.dict_words = {}
        self.childPar = []
        self.words_dict = words_dict
        self.pos_dict = pos_dict
        self.arc_dict = arc_dict
        self.unq_labels = list(labels)
        self.trans_labels = list(trans_dict.keys())
        self.arcs_labels = list(arc_dict.keys())
        self.model = model

    def parse(self):
        count = 0
        while (self.buffer or len(self.stack) > 1):
            feature_temp = self.getFeatures()
            feature_id = getWordPosArc(feature_temp, self.words_dict, self.pos_dict, self.arc_dict)
            feature_torch = torch.unsqueeze(torch.tensor(feature_id), dim=0)
            output_trans, output_arc = self.model(feature_torch)

            transition = self.trans_labels[torch.argmax(output_trans)]
            arc = self.arcs_labels[torch.argmax(output_arc)]

            if (count > 3 * len(self.sentence)):
                break

            # if (len(pred.split("_")) > 1):
            #     pred = pred.split("_")
            #     transition = pred[0]
            #     arc = pred[1]
            # else:
            #     transition = pred[0]
            #     arc = " "

            if self.buffer and len(self.stack)>1 and transition == "RIGHT":
                self.buffer[0].parId = self.stack[-1].id
                self.buffer[0].arc = arc

                self.addWord(self.buffer[0])
                self.rightArcShift()

            elif self.buffer and len(self.stack)>1 and transition == "LEFT":
                self.stack[-1].parId = self.buffer[0].id
                self.stack[-1].arc = arc
                self.addWord(self.stack[-1])
                self.leftArcShift()

            elif self.buffer and transition == "SHIFT":
                self.addWord(self.buffer[0])
                self.shift()

            elif len(self.stack)>1 and not self.buffer:
                self.reduce()
            count += 1

        return self.sentence

    def addWord(self, new_word):
        for i, word in enumerate(self.sentence):
            if word['id'] == new_word.id:
                word["parId"] = new_word.parId
                word["arc"] = new_word.arc
                # self.sentence[i] = new_word
                break

        # if word.id in self.dict_words:
        #     self.dict_words[word.id]=word
        # else:
        #     self.dict_words[word.id]=word

    def shift(self):
        self.stack.append(self.buffer.pop(0))

    def reduce(self):
        self.stack.pop()
        # else:
        #     self.stack.append(Word([0,None,None,None,'Root',None,-1,None,None,None]))

    def leftArcShift(self):
        self.buffer[0].childrens.append(self.stack[-1])
        self.stack.pop()

    def rightArcShift(self):
        self.stack[-1].childrens.append(self.buffer[0])
        self.stack.append(self.buffer.pop(0))

    def checkSizeAndAppend(self, arr):
        l = len(arr)
        # null_obj = wordObj([0, None, "Null_Word", None, "Null_Pos", None, -1, "Null_Arc"])

        null_obj = Word([0, None, "Null_Word", None, 'Null_Pos', None, -1, "Null_Arc", None, None])
        if l < 3:
            for i in range(3 - l):
                arr.append(null_obj)
        return arr

    def getFeatures(self):
        stack_len = len(self.stack)
        buffer_len = len(self.buffer)
        rev_stack = list(self.stack)
        if (rev_stack):
            rev_stack.pop(0)
        rev_stack.reverse()
        feature_temp = []
        buffer_temp = list(self.buffer)

        rev_stack = self.checkSizeAndAppend(rev_stack)
        buffer_temp = self.checkSizeAndAppend(buffer_temp)

        # get words and pos of first three words on stack and buffer
        feature_temp.extend(
            [rev_stack[0].lemma_word, rev_stack[1].lemma_word, rev_stack[2].lemma_word, rev_stack[0].finePos,
             rev_stack[1].finePos, rev_stack[2].finePos])
        feature_temp.extend(
            [buffer_temp[0].lemma_word, buffer_temp[1].lemma_word, buffer_temp[2].lemma_word, buffer_temp[0].finePos,
             buffer_temp[1].finePos, buffer_temp[2].finePos])

        rev_stack = checkChildrensAndAppend(rev_stack)

        # the words, POS tags, and arc labels of the first and second leftmost and rightmost children of the first two words on the stack
        # the words, POS tags, and arc labels of leftmost child of the leftmost child and rightmost
        # child of rightmost child of the first two words of the stack
        feature_temp.extend(getChildrenFeatures(rev_stack))
        self.features.append(feature_temp)

        # for i,item in enumerate(rev_stack):
        #     rev_stack[i].childrens = RemoveItems(rev_stack[i].childrens, None)

        return feature_temp

    def getTransactions(self):
        return self.arc_transitions

def RemoveItems(arr,item):
    res = list(filter((item).__ne__, arr))

    return res

def GetDataFromSentence(sentence):
    res=[]

    for word in sentence:
        dict_temp = dict(id=word.id, word=word.word,lemma_word=word.lemma_word,coarsePos=word.coarsePos,
                         finePos=word.finePos, morph=word.morph, parId=word.parId, arc=word.arc, other1=word.other1, other2 = word.other2)
        res.append(dict_temp)
    return res



if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description='Parse')

    argParser.add_argument('-m', type=str, help='model file')
    argParser.add_argument('-i', type=str, help='input file')
    argParser.add_argument('-o', type=str, help='output file')

    args = argParser.parse_args()

    parse = ParseFile(args.i)
    parse.readFile()
    data = parse.getSentences()


    vocab_dict = pickle.load(open("model_pikle", "rb"))
    model = pickle.load(open(args.m, "rb"))

    words_dict = vocab_dict["words_id"]
    pos_dict = vocab_dict["pos_id"]
    arc_dict = vocab_dict["arc_id"]
    unq_labels = vocab_dict["unique"]
    trans_dict = vocab_dict["transition_id"]

    model.eval()

    pred_features = []
    for i in trange(len(data)):
        sentence_dict=GetDataFromSentence(data[i])
        p = arcEagerShiftParser(sentence_dict,data[i], words_dict, pos_dict, arc_dict, unq_labels,trans_dict, model)
        pred_features.append(p.parse())

    write_file = WriteFile(pred_features, args.o)
    write_file.write()

    print("done")
