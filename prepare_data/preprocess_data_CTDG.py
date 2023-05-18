# coding=utf-8
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import pickle
import pdb
import random

def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])

      feat = np.array([float(x) for x in e[4:]])
      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u
    new_df.i = new_i

  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df

def build_time_edge_map(graph_df):
  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values
  time_set = list(set(timestamps))
  time_set.sort()

  time_edge_map = {}
  for tts in time_set:
    from_nodes = sources[timestamps == tts]
    to_nodes = destinations[timestamps == tts]
    edges = np.array([from_nodes, to_nodes])
    
    edges = edges.transpose(1,0)
    time_edge_map[tts] = edges

  return time_edge_map

def generate_init_graph(file_name, node_num):
  with open(file_name, 'w+') as f:
    for node_id in range(node_num):
      f.write('%d %d\n'%(node_id, node_id))
  print('init graph generate finish!!')

# generate train/val/test mask
def data_split(graph_df, split_file, different_new_nodes_between_val_and_test=False):
  print('Split data...')
  random.seed(2022)
  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  timestamps = graph_df.ts.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values

  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  # Compute nodes which appear at test time
  test_node_set = set(sources[timestamps > val_time]).union(
    set(destinations[timestamps > val_time]))
  # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

  # Mask saying for each source and destination whether they are new test nodes
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  # Mask which is true for edges with both destination and source not being new test nodes (because
  # we want to remove all edges involving any new test node)
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(sources[train_mask]).union(destinations[train_mask])
  assert len(train_node_set & new_test_node_set) == 0
  new_node_set = node_set - train_node_set

  if different_new_nodes_between_val_and_test:
    n_new_nodes = len(new_test_node_set) // 2
    val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
    test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

    edge_contains_new_val_node_mask = np.array(
      [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
    edge_contains_new_test_node_mask = np.array(
      [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
  else:
    edge_contains_new_node_mask = np.array(
        [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)
  np.savez(split_file, train_mask=train_mask,val_mask=val_mask,test_mask=test_mask,new_node_val_mask=new_node_val_mask,
    new_node_test_mask=new_node_test_mask, new_test_source_mask=new_test_source_mask, 
    new_test_destination_mask=new_test_destination_mask)
  print('Split data finish!!!')

  train_idx=np.where(train_mask==True)[0]
  val_idx=np.where(val_mask==True)[0]
  test_idx=np.where(test_mask==True)[0]
  ind_idx = np.concatenate((train_idx,val_idx,test_idx))
  u_list = list(sources[ind_idx])
  i_list = list(destinations[ind_idx])
  ts_list = list(timestamps[ind_idx])
  label_list = list(labels[ind_idx])
  idx_list = list(edge_idxs[ind_idx])

  ind_df = pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list})

  return ind_df

def run(data_name, bipartite=True, identity=True, inductive=False):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = '../data/{}.csv'.format(data_name)
  OUT_DF = '../data/{0}/ml_{0}.csv'.format(data_name)
  OUT_FEAT = '../data/{0}/ml_{0}.npy'.format(data_name)
  OUT_NODE_FEAT = '../data/{0}/ml_{0}_node.npy'.format(data_name)
  OUT_TIME_EDGE_MAP = '../data/{0}/{0}_time_edge_map.pkl'.format(data_name)
  OUT_INIT_GRAPH = '../data/{0}/{0}_init.txt'.format(data_name)

  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)
  print('reindex finish!!!')
  pdb.set_trace()

  if inductive:
    OUT_DATA_SPLIT = '../data/{0}/{0}_split_mask.npz'.format(data_name) #if inductive
    ind_df = data_split(new_df, OUT_DATA_SPLIT)
    pdb.set_trace()
    time_edge_map = build_time_edge_map(ind_df)
    OUT_TIME_EDGE_MAP = './data/{}_time_edge_map_inductive.pkl'.format(data_name)
  else:
    time_edge_map = build_time_edge_map(new_df)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  if identity:
    node_feat = np.eye(max_idx + 1)
  else:
    node_feat = np.zeros((max_idx + 1, 172))
  generate_init_graph(OUT_INIT_GRAPH, max_idx+1) # generate empty graph that contains only self-loop edges

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, node_feat)

  with open(OUT_TIME_EDGE_MAP, 'wb') as f:
    pickle.dump(time_edge_map, f, pickle.HIGHEST_PROTOCOL)
  print('save finish!!!')
  pdb.set_trace()

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
parser.add_argument('--identity', action='store_true', help='Whether the node feature is identity matrix')
parser.add_argument('--inductive', action='store_true', help='Whether process for inductive setting')

args = parser.parse_args()

run(args.data, bipartite=args.bipartite, identity=args.identity, inductive=args.inductive)

