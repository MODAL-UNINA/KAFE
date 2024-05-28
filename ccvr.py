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
from torch.utils.data import DataLoader, Dataset
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

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                inputs = batch['input_ids'].to(self.args.device)
                labels = batch['label'].to(self.args.device)
                net.zero_grad()

                logits, _ = net(inputs)
                loss = self.loss_func(logits, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


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



class CCVR(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.ldr_train = DataLoader(dataset, collate_fn=collate_fn_with_padding, batch_size=self.args.local_bs, shuffle=True)

    def client_ccvr_compute_feature_meanvar(self, net):
        net.train()
        all_class_features = {index: [] for index in range(self.args.num_classes)}
        for batch_idx, batch in enumerate(self.ldr_train):
            inputs = batch['input_ids'].to(self.args.device)
            labels = batch['label'].to(self.args.device)
            with torch.no_grad():
                _, feature = net(inputs)
                for sample_idx, label in enumerate(labels):
                    all_class_features[int(label)].append(feature[sample_idx])
        all_class_features = {key:val for key, val in all_class_features.items() if val != []}

        all_class_mean_feature = {index: [] for index in all_class_features.keys()}
        all_class_bias_feature = {index: [] for index in all_class_features.keys()}

        for label in all_class_features.keys():
            feature = torch.tensor([np.array(i.cpu().numpy()) for i in all_class_features[label]])
            if not torch.isnan(feature.std(dim=0)).any():
                all_class_bias_feature[label] = feature.std(dim=0)
                all_class_mean_feature[label] = torch.sum(feature, dim=0) / feature.shape[0]

        all_class_bias_feature = {key:val for key, val in all_class_bias_feature.items() if val != []}
        all_class_mean_feature = {key:val for key, val in all_class_mean_feature.items() if val != []}

        feature_meanvar = {'mean':all_class_mean_feature, 'var':all_class_bias_feature}

        return feature_meanvar


def freeze_layers(model, layers_to_freeze):
    for name, p in model.named_parameters():
        try:
            if name in layers_to_freeze:
                p.requires_grad = False
            else:
                p.requires_grad = True
        except:
            pass
    return model

class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.detach().float() 
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    def __len__(self):
        return self.images.shape[0]


def ccvr_func(args, central_node, local_meanvar_list):
    # compute global feature mean and var
    all_mean = []
    all_cov = []
    for idx in local_meanvar_list.keys():
        all_mean.append(local_meanvar_list[idx]['mean'])
        all_cov.append(local_meanvar_list[idx]['var'])

    global_mean_all = {index: [] for index in range(args.num_classes)}
    global_mean_avg = {index: [] for index in range(args.num_classes)}

    # compute mean for each class
    for client_num, feature_one in enumerate(all_mean):
        for class_num, feature in feature_one.items():
            global_mean_all[class_num].append(feature)
    # sum
    for index in range(args.num_classes):
        global_one_class = global_mean_all[index]
        # avoid divide by zero
        if len(global_one_class) != 0:
            global_mean_avg[index] = sum(global_one_class) / len(global_one_class)
        else:
            global_mean_avg[index] = torch.zeros((args.inputsize_classifier,))

    # compute covariance
    global_cov_all1 = {index: [] for index in range(args.num_classes)}
    global_cov_avg1 = {index: [] for index in range(args.num_classes)}
    # compute mean for each class
    for client_num, feature_one in enumerate(all_cov):
        for class_num, feature in feature_one.items():
            global_cov_all1[class_num].append(feature)
    # sum
    for index in range(args.num_classes):
        global_one_class = global_cov_all1[index]
        if len(global_one_class) != 0:
            global_cov_avg1[index] = sum(global_one_class) / len(global_one_class)
        else:
            global_cov_avg1[index] = torch.zeros((args.inputsize_classifier,))

    # the second part of the covariance
    global_cov_all2 = {index: [] for index in range(args.num_classes)}
    global_cov_avg2 = {index: [] for index in range(args.num_classes)}
    # compute mean for each class
    for client_num, feature_one in enumerate(all_mean):
        for class_num, feature in feature_one.items():
            global_cov_all2[class_num].append(feature * feature)
        # sum
    for index in range(args.num_classes):
        global_one_class = global_cov_all2[index]
        if len(global_one_class) != 0:
            global_cov_avg2[index] = sum(global_one_class) / len(global_one_class)
        else:
            global_cov_avg2[index] = torch.zeros((args.inputsize_classifier,))

    global_cov_avg3 = {index: [] for index in range(args.num_classes)}
    for index in range(args.num_classes):
        global_one_class1 = global_cov_avg1[index]
        global_one_class2 = global_cov_avg2[index]
        temp = global_one_class1 + global_one_class2
        avg_part3 = global_mean_avg[index]
        part3 = avg_part3 * avg_part3
        global_cov_avg3[index] = temp - part3

    # train the classifier
    global_mean = global_mean_avg
    global_cov = global_cov_avg3
    sample_num = [10] * args.num_classes

    # print(sum(sample_num))
    # sampling
    sampling_all = []
    label_all = []
    for i in range(args.num_classes):
        for _ in range(sample_num[i]):
            generate_sample = torch.normal(global_mean[i], global_cov[i]).to(args.device)
            sampling_all.append(generate_sample)
            label_one = torch.tensor(i).to(args.device)
            label_all.append(label_one)

    sampling_all = torch.stack(sampling_all, dim=0).to(args.device)
    label_all = torch.stack(label_all, dim=0).to(args.device)

    dst_train_syn_ft = TensorDataset(sampling_all, label_all)

    central_node = freeze_layers(central_node, ['embedding', 'lstm', 'linear', 'non_lin', 'dropout'])

    optimizer_ft_net = torch.optim.SGD(central_node.classifier.parameters(), lr=0.01, momentum=0.5)
    
    for epoch in range(10):
        trainloader_ft = torch.utils.data.DataLoader(dataset=dst_train_syn_ft,
                                    batch_size=128,
                                    shuffle=True)
        for data_batch in trainloader_ft:
            images, labels = data_batch
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = central_node.classifier(images)
            loss_net = F.cross_entropy(outputs, labels)
            optimizer_ft_net.zero_grad()
            loss_net.backward()
            optimizer_ft_net.step()

    return central_node

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


# model aggregation
def avg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# define args
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='ag_news', help="name of dataset")
parser.add_argument('--num_classes', type=int, default=4, help="number of classes")
parser.add_argument('--model', type=str, default='lstm_ag_news', help="model name")
parser.add_argument('--fl_method', type=str, default='ccvr', help="name of federated learning method")

parser.add_argument('--no_clients', default=30, help="number of clients: K")
parser.add_argument('--partition', type=str, default='dir', help="split method of data")
parser.add_argument('--alpha', type=float, default=1, help="parameter of dirichlet distribution")
parser.add_argument('--least_samples', type=int, default=100, help="the least samples of each client")

parser.add_argument('--rounds', type=int, default=200, help="rounds of communication")
parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
parser.add_argument('--bs', type=int, default=128, help="test batch size")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
parser.add_argument('--inputsize_classifier', type=int, default=256, help="input size of classifier, 256 for LSTM")

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


# dataset
if  args.dataset == 'ag_news':
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


# training
loss_train, acc_train = [], []
loss_test_list, acc_test_list, f1_test_list = [], [], []

start_time = time.time()

for iter in range(args.rounds):
    
    loss_locals = []
    w_locals = []

    # randomly select clients
    idxs_clients = np.random.choice(range(args.no_clients), 10, replace=False)

    clients_trn_data = {i: len(clients_total[i]) for i in idxs_clients}
    # calculate the weighting factor of clients
    weight_scalling_factor_list = [clients_trn_data[i] for i in idxs_clients]/np.sum([clients_trn_data[i] for i in idxs_clients])


    for idx in idxs_clients:
        local = LocalUpdate(args, dataset=clients_total[idx])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

        w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))

    # update global weights by weight average
    w_glob = avg(w_locals)

    # copy weight to net_glob
    net_glob.load_state_dict(w_glob)

    # compute mean and variance of the feature
    local_meanvar_list = {}
    for idx in idxs_clients:
        ccvr = CCVR(args, dataset=clients_total[idx])
        feature_meanvar = ccvr.client_ccvr_compute_feature_meanvar(net_glob)
        local_meanvar_list[idx] = copy.deepcopy(feature_meanvar)

    # retrain the classifier
    net_glob = ccvr_func(args, net_glob, local_meanvar_list)

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

