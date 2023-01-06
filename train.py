import torch
import numpy as np
from tqdm import tqdm, trange
from torch import nn
from model import ArcEagerParser
from torch.utils.data import TensorDataset, DataLoader
import argparse
import pickle


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
        # if dict[word]:
        #     res.append(dict[word])
        # else:
        #     res.append(dict["UNK"])
        res.append(dict.get(word, dict["UNK"]))
    return res


def getDataFromFile(file, word_dict, pos_dict, arc_dict, transition_dict):
    features = []
    # labels = []
    arcs =[]
    transitions =[]

    f = open(file, encoding="utf8")
    for line in tqdm(f):
        data = line.strip().split('\t')
        if len(data[0].split("_")) > 1:
            temp = data[0].split("_")
            transitions.append(transition_dict[temp[0]])
            arcs.append(arc_dict[temp[1]])
            # labels.append([transition_dict[temp[0]], arc_dict[temp[1]] ])
        else:
            transitions.append( transition_dict[data[0]] )
            arcs.append(arc_dict[" "])
            # labels.append([transition_dict[data[0]], " "])

        # labels.append(label_dict.get(data[0], "UNK"))
        features.append(getWordPosArc(data[1:], word_dict, pos_dict, arc_dict))

    return features, transitions, arcs


def accuracy(y_pred, y_true):
    output = torch.argmax(y_pred, dim=1)

    true_value = (output == y_true).float()
    acc = true_value.sum() / len(true_value)

    return acc.tolist()


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, x, y_trans, y_arcs):
        self.x = x
        self.y_trans = y_trans
        self.y_arcs = y_arcs

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return torch.tensor(self.x[index]), torch.tensor(self.y_trans[index]), torch.tensor(self.y_arcs[index])


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"training on device {device}")
    # transitions = ["SHIFT","REDUCE", "LEFT", "RIGHT"]
    # transitions2id = {w:k for (k,w) in enumerate(transitions)}
    batch_size = args.b
    hidden_units = args.u
    lr = args.l
    epochs = args.e
    model_file = args.o
    input_file = args.i
    val_file = args.v
    model_file = args.o

    vocab_dict = pickle.load(open("model_pikle", "rb"))

    words_size = len(vocab_dict["words_id"])
    pos_size = len(vocab_dict["pos_id"])
    arc_size = len(vocab_dict["arc_id"])
    trans_size = len(vocab_dict["transition_id"])

    unq_label_size = len(vocab_dict["unique"])
    labels = list(vocab_dict["unique"])
    labels_id = {w: k for (k, w) in enumerate(vocab_dict["unique"])}

    x_train, y_train_transtn, y_train_arcs = getDataFromFile(input_file, vocab_dict["words_id"], vocab_dict["pos_id"], vocab_dict["arc_id"],
                                       vocab_dict["transition_id"])
    x_val, y_val_transtn, y_val_arcs = getDataFromFile(val_file, vocab_dict["words_id"], vocab_dict["pos_id"], vocab_dict["arc_id"],
                                   vocab_dict["transition_id"])

    train_data = TextDataset(x_train, y_train_transtn, y_train_arcs)
    val_data = TextDataset(x_val, y_val_transtn, y_val_arcs)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    model = ArcEagerParser(hidden_units, words_size, pos_size, arc_size, trans_size)
    model.to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=1e-8)
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    trange_epochs = trange(epochs)
    for i in (trange_epochs):
        model.train()

        train_loss = 0
        val_loss = 0

        train_acc = 0
        val_acc = 0

        for xtrain, ytrain_trans, ytrain_arc in (train_loader):
            optimizer.zero_grad()
            outputs_trans, output_arc = model(xtrain.to(device))

            loss_trans = loss_fn(outputs_trans.float(), ytrain_trans.to(device))
            loss_arcs = loss_fn(output_arc.float(), ytrain_arc.to(device))
            loss_final = loss_arcs+loss_trans
            loss_final.backward()

            optimizer.step()

            train_loss += loss_trans.detach().cpu().item()
            train_loss+= loss_arcs.detach().cpu().item()

            train_acc += accuracy(outputs_trans.to(device), ytrain_trans.to(device))/2
            train_acc += accuracy(output_arc.to(device), ytrain_arc.to(device))/2


        train_accuracies.append(train_acc / len(train_loader))
        train_losses.append(train_loss / len(train_loader))
        model.eval()

        with torch.no_grad():
            for xval, yval_trans, yval_arcs in val_loader:
                outputs_trans, output_arc = model(xval.to(device))

                loss_trans = loss_fn(outputs_trans.float(), yval_trans.to(device))
                loss_arcs = loss_fn(output_arc.float(), yval_arcs.to(device))

                val_acc += accuracy(outputs_trans.to(device), yval_trans.to(device))/2
                val_acc += accuracy(output_arc.to(device), yval_arcs.to(device))/2


            val_accuracies.append(val_acc / len(val_loader))
            val_losses.append(val_loss / len(val_loader))

        if i % 2 == 0:
            # print(f"Training loss:{sum(train_losses)/len(train_losses)}")
            trange_epochs.set_description(
                f"Training loss:{sum(train_losses) / len(train_losses)} | Validation loss: {sum(val_losses) / len(val_losses)} | Training Accuracy:{sum(train_accuracies) / len(train_accuracies)} | Validation Accuracy: {sum(val_accuracies) / len(val_accuracies)}  ")
            # trange_epochs.set_description(f"Training Accuracy:{sum(train_accuracies)/len(train_accuracies)} | Validation Accuracy: {sum(val_accuracies)/len(val_accuracies) } ")

    # torch.save(model.state_dict(), model_file)
    with open("model_256.train", 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural net training arguments.')

    parser.add_argument('-u', type=int, help='number of hidden units')
    parser.add_argument('-l', type=float, help='learning rate')
    parser.add_argument('-b', type=int, help='mini-batch size')
    parser.add_argument('-e', type=int, help='number of epochs to train for')
    parser.add_argument('-i', type=str, help='training file')
    parser.add_argument('-v', type=str, help='validation file')
    parser.add_argument('-o', type=str, help='model file to be written')

    args = parser.parse_args()

    train(args)

