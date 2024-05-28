import psutil
import time
import random
import copy
import argparse
import sys
import numpy as np
import torch
from torch import nn
from torchvision import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PatchCollection
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import pickle
import datasets
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter
from typing import List
import string
nltk.download('punkt')



_interactive_mode = 'ipykernel_launcher' in sys.argv[0] or \
                    (len(sys.argv) == 1 and sys.argv[0] == '')

if _interactive_mode:
    from tqdm.auto import tqdm, trange
else:
    from tqdm import tqdm, trange

def is_interactive():
    return _interactive_mode

def myshow(plot_show=True):
    if _interactive_mode and plot_show:
        plt.show()
    else:
        plt.close()


# function for non-iid data split
def separate_data(data, num_clients, num_classes, least_samples=None, partition=None, alpha=0.1):
    X = {}
    y = {}
    statistic = {}

    dataset_content, dataset_label = data

    dataidx_map = {}

    if partition == "dir":

        min_size = 0
        K = num_classes
        N = len(dataset_label)

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        statistic.setdefault(client, [])
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs].copy()
        y[client] = dataset_label[idxs].copy()

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
        
    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)
    print("Separate data finished!\n")

    ylabels = np.arange(num_clients)
    xlabels = np.arange(num_classes)

    x_mesh, y_mesh = np.meshgrid(np.arange(num_classes), np.arange(num_clients))

    s = np.zeros((num_clients,num_classes), dtype=np.uint16)
    for k_stat, v_stat in statistic.items():
        for elem in v_stat:
            s[k_stat, elem[0]] = elem[1]

    c = np.ones((num_clients,num_classes), dtype=np.uint16)
    R = s/s.max()/3

    viridis_cmap = cm.get_cmap('viridis')

    new_start = 0.5
    new_end = 1.0

    cm_color = cm.colors.LinearSegmentedColormap.from_list(
        'viridis_cut', viridis_cmap(np.linspace(new_start, new_end, 256)))

    fig, ax = plt.subplots(figsize=(0.4*num_clients, 0.4*num_classes)) #, facecolor='lightgrey')
    circles = [plt.Circle((i,j), radius=r) for r, j, i in zip(R.flat, x_mesh.flat, y_mesh.flat)]
    col = PatchCollection(circles, array=c.flatten(), cmap=cm_color, zorder=10) # cmap="RdYlGn")
    ax.add_collection(col)
    ax.set(title='Number of samples per class per client')
    ax.set(xlabel='Client ID', ylabel='Classes')
    ax.set(xticks=np.arange(num_clients), yticks=np.arange(num_classes),
        xticklabels=ylabels, yticklabels=xlabels)
    ax.set_xticks(np.arange(num_clients+1)-0.5, minor=True)
    ax.set_yticks(np.arange(num_classes+1)-0.5, minor=True)
    # plt.xticks(rotation=70)
    ax.grid(which='minor', zorder=0, alpha=0.5, color='white')
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_facecolor("#E8EAED")
    myshow()

    return X, y, statistic


class WordDataset:
    def __init__(self, sentences):
        self.data = sentences
        self.unk_id = word2ind['<unk>']
        self.bos_id = word2ind['<bos>']
        self.eos_id = word2ind['<eos>']
        self.pad_id = word2ind['<pad>']

    def __getitem__(self, idx: int) -> List[int]:
        processed_text = self.data[idx]['text'].lower().translate(
            str.maketrans('', '', string.punctuation))
        tokenized_sentence = [self.bos_id]
        tokenized_sentence += [
            word2ind.get(word, self.unk_id) for word in word_tokenize(processed_text)
            ] 
        tokenized_sentence += [self.eos_id]

        train_sample = {
            "text": tokenized_sentence,
            "label": self.data[idx]['label']
        }

        return train_sample

    def __len__(self) -> int:
        return len(self.data)


class LocalUpdate(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index=word2ind['<pad>'])
        self.ldr_train = DataLoader(dataset, collate_fn=collate_fn_with_padding, batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, global_protos):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            agg_protos_label = {}

            for batch_idx, batch in enumerate(self.ldr_train):
                inputs = batch['input_ids'].to(self.args.device)
                labels = batch['label'].to(self.args.device)
                net.zero_grad()

                logits, protos = net(inputs)
                loss1 = self.loss_func(logits, labels)

                loss_mse = nn.MSELoss()

                if len(global_protos) == 0:
                    loss2 = 0*loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)

                loss = loss1 + loss2 * args.ld

                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if labels[i].item() in agg_protos_label:
                        agg_protos_label[labels[i].item()].append(protos[i,:])
                    else:
                        agg_protos_label[labels[i].item()] = [protos[i,:]]

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), agg_protos_label    


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(11842, 256)
        self.lstm = nn.LSTM(256, 256, 2, batch_first=True)
        self.linear = nn.Linear(256, 256)
        self.projection = nn.Linear(256, args.num_classes)

        self.non_lin = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x) -> torch.Tensor:
        embeddings = self.embedding(x)
        output, _ = self.lstm(embeddings)

        output = output.max(dim=1)[0]

        x = self.non_lin(output)
        x = self.linear(x)
        
        x = self.dropout(x)

        x = self.non_lin(x)
        prediction = self.projection(x)

        return prediction, x


def test_text(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0

    all_pred = []
    all_target = []

    data_loader = DataLoader(datatest, collate_fn=collate_fn_with_padding, batch_size=args.bs)
    l = len(data_loader)
    for idx, batch in enumerate(data_loader):
        if args.gpu != -1:
            inputs, target = batch['input_ids'].to(args.device), batch['label'].to(args.device)
        logits, _ = net_g(inputs)
        # sum up batch loss
        test_loss += F.cross_entropy(logits, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = logits.argmax(dim=1, keepdim=True)
        correct += y_pred.eq(target.view_as(y_pred)).sum().item()

        all_pred.append(y_pred)
        all_target.append(target)

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    # f1_score
    all_pred = torch.cat(all_pred, dim=0).cpu().numpy()
    all_target = torch.cat(all_target, dim=0).cpu().numpy()
    f1 = f1_score(all_target, all_pred, average='macro') * 100
    
    print('\nTest accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(data_loader.dataset), accuracy))
    
    return accuracy, test_loss, f1


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label


# define args
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='ag_news', help="name of dataset")
parser.add_argument('--num_classes', type=int, default=4, help="number of classes")
parser.add_argument('--model', type=str, default='lstm_ag_news', help="model name")
parser.add_argument('--fl_method', type=str, default='fedproto-average', help="name of federated learning method")

parser.add_argument('--no_clients', default=30, help="number of clients: K")
parser.add_argument('--partition', type=str, default='dir', help="split method of data")
parser.add_argument('--alpha', type=float, default=1, help="parameter of dirichlet distribution")
parser.add_argument('--least_samples', type=int, default=100, help="the least samples of each client")

parser.add_argument('--ld', type=float, default=1, help="weight of proto loss")
parser.add_argument('--rounds', type=int, default=200, help="rounds of communication")
parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
parser.add_argument('--bs', type=int, default=128, help="test batch size")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
parser.add_argument('--seed', type=int, default=0, help="random seed (default: 0)")

args = parser.parse_args("")

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

# set seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


if args.dataset == 'ag_news':
    # for tokenization
    dataset = datasets.load_dataset('ag_news')
    words = Counter()

    for example in dataset['train']['text']:
        # Lowercase and remove punctuation
        prccessed_text = example.lower().translate(
            str.maketrans('', '', string.punctuation))

        for word in word_tokenize(prccessed_text):
            words[word] += 1


    vocab = set(['<unk>', '<bos>', '<eos>', '<pad>'])
    counter_threshold = 25

    for char, cnt in words.items():
        if cnt > counter_threshold:
            vocab.add(char)

    print(f'Vocab size: {len(vocab)}')

    word2ind = {char: i for i, char in enumerate(vocab)}
    ind2word = {i: char for char, i in word2ind.items()}

    def collate_fn_with_padding(
        input_batch: List[List[int]], pad_id=word2ind['<pad>']) -> torch.Tensor:
        seq_lens = [len(x['text']) for x in input_batch]
        max_seq_len = min(max(seq_lens), 256)

        new_batch = []
        for sequence in input_batch:
            sequence['text'] = sequence['text'][:max_seq_len]
            for _ in range(max_seq_len - len(sequence['text'])):
                sequence['text'].append(pad_id)

            new_batch.append(sequence['text'])
        
        sequences = torch.LongTensor(new_batch).to(args.device)
        labels = torch.LongTensor([x['label'] for x in input_batch]).to(args.device)

        new_batch = {
            'input_ids': sequences,
            'label': labels}

        return new_batch

    # load dataset and split
    dataset_train = dataset['train']
    dataset_test = dataset['test']

    dataset_content = np.array([x['text'] for x in dataset_train])
    dataset_label = np.array([x['label'] for x in dataset_train])

    dataset_test = WordDataset([{'text': x['text'], 'label': x['label']} for x in dataset_test])
                                
else:
    raise ValueError('Dataset Error')


X, y, statistic = separate_data((dataset_content, dataset_label), args.no_clients, args.num_classes, least_samples=args.least_samples, partition=args.partition, alpha=args.alpha)


clients_total = {}
for i in range(args.no_clients):
    texts = X[i]
    labels = y[i]
    clients_total[i] = WordDataset([{'text': text, 'label': label} for text, label in zip(texts, labels)])


net_glob = LSTM(args=args).to(args.device)
net_glob.train()

local_model_list = {}

# copy global model to local model, initialize
for i in range(args.no_clients):
    local_model_list[i] = copy.deepcopy(net_glob)


# training
loss_train, acc_train = [], []
loss_test_list, acc_test_list, f1_test_list = [], [], []

start_time = time.time()

global_protos = []

for iter in range(args.rounds):
    
    loss_locals = []
    w_locals = []
    local_protos = {}
    
    # randomly select clients
    idxs_clients = np.random.choice(range(args.no_clients), 10, replace=False)

    for idx in idxs_clients:
        local = LocalUpdate(args=args, dataset=clients_total[idx])
        w, loss, protos = local.train(net=local_model_list[idx], global_protos=global_protos)
        agg_protos = agg_func(protos)
        
        w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))
        local_protos[idx] = agg_protos


    # update global weights
    w_glob = average_weights(w_locals)

    net_glob.load_state_dict(w_glob, strict=True)

    # update local weights list
    for idx in idxs_clients:
        local_model_list[idx].load_state_dict(w_glob, strict=True)

    # update global prototype
    global_protos = proto_aggregation(local_protos)

    # print train loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Average Train loss {:.3f}'.format(iter, loss_avg))
    loss_train.append(loss_avg)     # collect train loss

    # evaluate on test set
    acc_test, loss_test, f1 = test_text(net_glob, dataset_test, args)

    acc_test_list.append(acc_test)
    loss_test_list.append(loss_test)
    f1_test_list.append(f1)
    

end_time = time.time()
cost_time = end_time - start_time

memory_used = psutil.Process().memory_info().rss
memory_cost = memory_used / 1024 / 1024

print("Total time cost: ", cost_time)
print("Total memory cost: ", memory_cost)

# save accuracy
with open('../results/ag_news/seed{}_{}_test_acc.pkl'.format(args.seed, args.fl_method), 'wb') as f:
    pickle.dump({'acc_test_list': acc_test_list}, f)
