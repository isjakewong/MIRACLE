from __future__ import division
from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
torch.set_printoptions(threshold=1000000)
import torch.nn as nn
import numpy as np
import random
np.set_printoptions(threshold=10e6)
import pandas as pd
import scipy.sparse as sp
from argparse import ArgumentParser, Namespace
import os
import pickle
import networkx as nx
import time
from typing import Union, Tuple, List, Dict
np.set_printoptions(threshold=np.inf)
from global_graph.utils import mask_test_edges, sparse_to_tuple, sparse_mx_to_torch_sparse_tensor, \
    normalize_adj
from global_graph.metrics import get_roc_score
from global_graph.model_hier import HierGlobalGCN
from global_graph.utils import save_checkpoint, load_checkpoint
from utils import create_logger, get_available_pretrain_methods, get_available_data_types
from features import get_features_generator, get_available_features_generators

def parse_args():
    parser = ArgumentParser()
    # data
    parser.add_argument('--data_path', type=str, default='datachem/ZhangDDI_train.csv', choices=['datachem/ZhangDDI_train.csv', 'datachem/ChChMiner_train.csv', 'datachem/DeepDDI_train.csv'])
    parser.add_argument('--separate_val_path', type=str, default='datachem/ZhangDDI_valid.csv', choices=['datachem/ZhangDDI_valid.csv', 'datachem/ChChMiner_valid.csv', 'datachem/DeepDDI_valid.csv'])
    parser.add_argument('--separate_test_path', type=str, default='datachem/ZhangDDI_test.csv', choices=['datachem/ZhangDDI_test.csv', 'datachem/ChChMiner_test.csv', 'datachem/DeepDDI_test.csv'])
    parser.add_argument('--vocab_path', type=str, default='datachem/drug_list_zhang.csv', choices=['datachem/drug_list_zhang.csv', 'datachem/drug_list_miner.csv', 'datachem/drug_list_deep.csv'])

    # training
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=170, help='Number of epochs to train.')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--gpu', type=int, default=1)

    # architecture
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--num_heads1', type=int, default=1)
    parser.add_argument('--num_heads2', type=int, default=3)

    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Undirected edges (always sum the two relevant bond vectors)')
    parser.add_argument('--output_size', type=int, default=1,
                        help='output dim for higher-capacity FFN')
    parser.add_argument('--ffn_hidden_size', type=int, default=264,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=3,
                        help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='Use messages on atoms instead of messages on bonds')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network')
    parser.add_argument('--weight_tying', action='store_false', default=True)
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--attn_output', action='store_true', default=True)
    parser.add_argument('--attn_num_d', type=int, default=30, help='Number of units in attention weight parameters 1.')
    parser.add_argument('--attn_num_r', type=int, default=10, help='Number of units in attention weight parameters 2.')

    parser.add_argument('--FF_hidden1', type=int, default=300, help='Number of units in FF hidden layer 1.')
    parser.add_argument('--FF_hidden2', type=int, default=281, help='Number of units in FF hidden layer 2.')
    parser.add_argument('--FF_output', type=int, default=264, help='Number of units in FF hidden layer 3.')
    parser.add_argument('--gcn_hidden1', type=int, default=300, help='Number of units in GCN hidden layer 1.')
    parser.add_argument('--gcn_hidden2', type=int, default=281, help='Number of units in GCN hidden layer 2.')
    parser.add_argument('--gcn_hidden3', type=int, default=264, help='Number of units in GCN hidden layer 3.')
    parser.add_argument('--bias', action='store_true', default=True)
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--clip', type=float, default=1.0, help='clip coefficient')

    parser.add_argument('--smiles_based', action='store_true', default=False)
    parser.add_argument('--graph_encoder', type=str, default='dmpnn', choices=['dmpnn', 'ggnn'])
    parser.add_argument('--jt', action='store_true', default=False, help='only use junction tree (default: false)')
    parser.add_argument('--pretrain', type=str, choices=get_available_pretrain_methods(), default='mol2vec')
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--pooling', type=str, choices=['max', 'sum', 'lstm'], default='sum')
    parser.add_argument('--emb_size', type=int, default=None)

    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--radius', type=int, default=1)
    parser.add_argument('--data_type', type=str, choices=get_available_data_types(), default='small')
    parser.add_argument('--min_freq', type=int, default=3)
    parser.add_argument('--num_edges_w', type=float, default=7000)
    parser.add_argument('--alpha_loss', type=float, default=1)
    parser.add_argument('--beta_loss', type=float, default=0.8)
    parser.add_argument('--gamma_loss', type=float, default=1)
    
    # store
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--prior', dest='prior', action='store_const', const=True, default=True)  
    parser.add_argument('--use_input_features_generator', type=str, default=None, choices=get_available_features_generators())   
    parser.add_argument('--input_features_size', type=int, default=0, help='Number of input features dimension.')
    args = parser.parse_args()
    # modify
    args.use_input_features = args.use_input_features_generator in get_available_features_generators()
    if args.vocab_path == 'datachem/drug_list_deep.csv':
        args.learning_rate = 0.0001
    if args.use_input_features:
        args.input_features_size = 512
    return args


def seed_everything():
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    torch.backends.cudnn.deterministic = True


def load_vocab(filepath: str):
    df = pd.read_csv(filepath, index_col=False)
    smiles2id = {smiles: idx for smiles, idx in zip(df['smiles'], range(len(df)))}
    return smiles2id


def load_csv_data(filepath: str, smiles2id: dict, is_train_file: bool = True) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    df = pd.read_csv(filepath, index_col=False)

    edges = []
    edges_false = []
    for row_id, row in df.iterrows():
        row_dict = dict(row)
        smiles_1 = row_dict['smiles_1']
        smiles_2 = row_dict['smiles_2']
        if smiles_1 in smiles2id.keys() and smiles_2 in smiles2id.keys():
            idx_1 = smiles2id[smiles_1]
            idx_2 = smiles2id[smiles_2]
            label = int(row_dict['label'])
        else:
            continue
        if label > 0:
            edges.append((idx_1, idx_2))
            edges.append((idx_2, idx_1))
        else:
            edges_false.append((idx_1, idx_2))
            edges_false.append((idx_2, idx_1))
    if is_train_file:
        edges = np.array(edges, dtype=np.int)
        edges_false = np.array(edges_false, dtype=np.int)
        return edges, edges_false
    else:
        edges = np.array(edges, dtype=np.int)
        edges_false = np.array(edges_false, dtype=np.int)
        return edges, edges_false

def load_data(args: Namespace, filepath: str, smiles2idx: dict = None) \
        -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray,
                 np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ext = os.path.splitext(filepath)[-1]
    if args.separate_val_path is not None and args.separate_test_path is not None and ext == '.csv':
        """
        .csv file can only provide (node1, node2) edges
        1. load vocab file
        2. load edges
        3. construct adj 
        """
        assert smiles2idx is not None
        num_nodes = len(smiles2idx)
        train_edges, train_edges_false = load_csv_data(filepath, smiles2idx, is_train_file=True)
        val_edges, val_edges_false = load_csv_data(args.separate_val_path, smiles2idx, is_train_file=False)
        test_edges, test_edges_false = load_csv_data(args.separate_test_path, smiles2idx, is_train_file=False)

        all_edges = np.concatenate([train_edges, val_edges, test_edges], axis=0)
        data = np.ones(all_edges.shape[0])

        adj = sp.csr_matrix((data, (all_edges[:, 0], all_edges[:, 1])),
                            shape=(num_nodes, num_nodes))

        data_train = np.ones(train_edges.shape[0])
        data_train_false = np.ones(train_edges_false.shape[0])
        
        adj_train = sp.csr_matrix((data_train, (train_edges[:, 0], train_edges[:, 1])),
                                  shape=(num_nodes, num_nodes))
        adj_train_false = sp.csr_matrix((data_train_false, (train_edges_false[:, 0], train_edges_false[:, 1])), \
                                              shape=(num_nodes, num_nodes))

        return adj, adj_train, adj_train_false, train_edges, train_edges_false,\
               val_edges, val_edges_false, \
               test_edges, test_edges_false


def select_features(args: Namespace, idx2smiles: List[str] = None, num_nodes: int = None) \
        -> Union[sp.dia_matrix, List[str], np.ndarray]:
    if args.use_input_features:
        assert idx2smiles is not None
        num_nodes = len(idx2smiles)
        fg_func = get_features_generator(args.use_input_features_generator)
        try:
            num_features = fg_func(idx2smiles[0]).shape[0]
        except AttributeError:
            num_features = np.array(fg_func(idx2smiles[0])).shape[0]
        features = np.zeros((num_nodes, num_features), dtype=np.float)
        for smiles_idx, smiles in enumerate(idx2smiles):
            features[smiles_idx, :] = fg_func(smiles)
        return idx2smiles, features
    else:
        return idx2smiles

args = parse_args()
logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
seed_everything()
args.cuda = True if torch.cuda.is_available() else False
if args.cuda:
    torch.cuda.set_device(args.gpu)

def main():
    smiles2idx = load_vocab(args.vocab_path) if args.vocab_path is not None else None
    if smiles2idx is not None:
        idx2smiles = [''] * len(smiles2idx)
        for smiles, smiles_idx in smiles2idx.items():
            idx2smiles[smiles_idx] = smiles
    else:
        idx2smiles = None
    adj, adj_train, adj_train_false, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        load_data(args, filepath=args.data_path, smiles2idx=smiles2idx)
    num_nodes = adj.shape[0]
    num_edges = adj.sum()

    logger.info('Number of nodes: {}, number of edges: {}'.format(num_nodes, num_edges))
    if args.use_input_features:
        features_orig, features_generated = select_features(args, idx2smiles, num_nodes)
    else:
        features_orig = select_features(args, idx2smiles, num_nodes)

    num_features = args.hidden_size 
    features_nonzero = 0
    args.num_features = num_features
    args.features_nonzero = features_nonzero

    # input for model
    adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_orig.eliminate_zeros()
    adj = adj_train

    num_edges_w = adj.sum()
    num_nodes_w = adj.shape[0]
    args.num_edges_w = num_edges_w
    pos_weight = float(num_nodes_w ** 2 - num_edges_w) / num_edges_w

    adj_tensor = torch.FloatTensor(adj.toarray())
    drug_nums = adj.toarray().shape[0]
    args.drug_nums = drug_nums

    adj_norm = normalize_adj(adj)

    adj_label = adj_train
    adj_mask = pos_weight * adj_train.toarray() + adj_train_false.toarray()
    adj_mask_un = adj.toarray() - adj_train.toarray() - adj_train_false.toarray()
    adj_mask = torch.flatten(torch.Tensor(adj_mask))
    adj_mask_un = torch.flatten(torch.Tensor(adj_mask_un))

    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_label)
    features = features_orig

    if args.cuda:
        adj_norm = adj_norm.cuda()
        adj_label = adj_label.cuda()
        adj_tensor = adj_tensor.cuda()
        adj_mask = adj_mask.cuda()
        adj_mask_un = adj_mask_un.cuda()

    logger.info('Create model and optimizer')
    model = HierGlobalGCN(args, num_features, features_nonzero,
                            dropout=args.dropout,
                            bias=args.bias,
                            sparse=False)

    if args.cuda:
        model.cuda()

    loss_function_BCE = nn.BCEWithLogitsLoss(reduction='none')
    loss_function_KL = nn.KLDivLoss(reduction='none')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    logger.info('Train model')
    best_score, best_epoch = 0, 0

    for epoch in range(args.epochs):
        t = time.time()

        model.train()
        optimizer.zero_grad()

        preds, feat, embed, re_feat, re_embed, DGI_loss = model(features, adj_norm, adj_tensor, drug_nums)
        labels = adj_label.to_dense().view(-1)

        re_embed_pos = torch.log(re_embed)
        BCEloss = torch.mean(loss_function_BCE(preds, labels) * adj_mask)
        BCEloss += torch.mean(loss_function_BCE(re_feat, labels) * adj_mask)
        KLloss = torch.mean(loss_function_KL(re_embed_pos, re_feat) * adj_mask)

        avg_cost = args.alpha_loss * BCEloss + args.beta_loss * KLloss + args.gamma_loss * DGI_loss
        roc_curr, ap_curr, f1_curr, acc_score = get_roc_score(
            model, features, adj_norm, adj_orig, adj_tensor, drug_nums, val_edges, val_edges_false
        )

        logger.info('Epoch: {} train_loss= {:.5f} val_roc= {:.5f} val_ap= {:.5f}, val_f1= {:.5f}, val_acc={:.5f}, time= {:.5f}'.format(
            epoch + 1, avg_cost, roc_curr, ap_curr, f1_curr, acc_score, time.time() - t,
        ))
        if roc_curr > best_score:
            best_score = roc_curr
            best_epoch = epoch
            if args.save_dir:
                save_checkpoint(os.path.join(args.save_dir, 'model.pt'), model, args)

        # update parameters
        avg_cost.backward()
        if args.vocab_path == 'datachem/drug_list_deep.csv':
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
 
    logger.info('Optimization Finished!')

    # Evaluate on test set using model with best validation score
    if args.save_dir:
        logger.info(f'Model best validation roc_auc = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(args.save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

    roc_score, ap_score, f1_score, acc_score = get_roc_score(
        model, features, adj_norm, adj_orig, adj_tensor, drug_nums, test_edges, test_edges_false, test = True
    )
    logger.info('Test ROC score: {:.5f}'.format(roc_score))
    logger.info('Test AP score: {:.5f}'.format(ap_score))
    logger.info('Test F1 score: {:.5f}'.format(f1_score))
    logger.info('Test ACC score: {:.5f}'.format(acc_score))

if __name__ == '__main__':
    main()
