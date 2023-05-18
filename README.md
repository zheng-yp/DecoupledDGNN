# DecoupledDGNN
Repository for code to reproduce the paper Decoupled Graph Neural Networks for Large Dynamic Graphs

## Requirements
- CUDA 10.1
- python 3.8.5
- pytorch 1.7.1
- GCC 5.4.0
- cython 0.29.21
- eigency 1.77
- numpy 1.18.1
- torch-geometric 1.6.3 
- tqdm 4.56.0
- ogb 1.2.4
- [eigen 3.3.9] (https://gitlab.com/libeigen/eigen.git)

## Compilation

Cython needs to be compiled before running, run this command:
```
python setup.py build_ext --inplace
```

## Preprocessing data
```
python preprocess_data_CTDG.py --data wikipedia --bipartite
python preprocess_data_CTDG.py --data reddit --bipartite
```

## Gen temporal representation
```
### Wikipedia
python gen_emb.py --path ./data/wikipedia/ --data wikipedia --randomize_features --undirect
### Reddit
python gen_emb.py --path ./data/reddit/ --data reddit --randomize_features --undirect
### UCI-MSG
python gen_emb.py --path ./data/CollegeMsg/ --data CollegeMsg --randomize_features --disperse
### Bitcoin-Alpha
python gen_emb.py --data bitcoinalpha --path ./data/bitcoinalpha/ --randomize_features --disperse
### Bitcoin-OTC
python gen_emb.py --data bitcoinotc --path ./data/bitcoinotc/ --randomize_features --disperse
### GDELT
python gen_emb.py --path ./data/GDELT/ --data GDELT
```

## Furture Link Prediction - CTDG
```
### Wikipedia
python link_prediction.py --data wikipedia --emb_size 172 --batch_size 128 --epochs 30 --learning_rate 0.0001 --hidden_dim 128 --patience 30 --gpu 1 --window_size 20 --seq_model lstm
### Reddit
python link_prediction.py --data reddit --emb_size 172 --batch_size 128 --epochs 30 --learning_rate 0.0001 --hidden_dim 128 --patience 30 --gpu 1 --window_size 20 --seq_model lstm
```

## Furture Link Prediction - DTDG
```
### UCI-MSG
python link_prediction_alongtime.py --data CollegeMsg --emb_size 128 --batch_size 1024 --epochs 30 --learning_rate 0.001 --hidden_dim 64 --patience 30 --gpu 1 --window_size 20 --seq_model lstm
python link_prediction_alongtime.py --data CollegeMsg --emb_size 128 --batch_size 1024 --epochs 30 --learning_rate 0.001 --hidden_dim 64 --patience 30 --gpu 1 --window_size 20 --seq_model gru
python link_prediction_alongtime.py --data CollegeMsg --emb_size 128 --batch_size 1024 --epochs 30 --learning_rate 0.001 --hidden_dim 64 --patience 30 --gpu 1 --window_size 20 --seq_model transformer


### Bitcoin-Alpha
python link_prediction_alongtime.py --data bitcoinalpha --emb_size 128 --batch_size 1024 --epochs 30 --learning_rate 0.001 --hidden_dim 128 --patience 30 --gpu 1 --window_size 20 --seq_model lstm
python link_prediction_alongtime.py --data bitcoinalpha --emb_size 128 --batch_size 1024 --epochs 30 --learning_rate 0.001 --hidden_dim 128 --patience 30 --gpu 1 --window_size 20 --seq_model gru
python link_prediction_alongtime.py --data bitcoinalpha --emb_size 128 --batch_size 1024 --epochs 30 --learning_rate 0.001 --hidden_dim 128 --patience 30 --gpu 1 --window_size 20 --seq_model transformer

### Bitcoin-OTC
python link_prediction_alongtime.py --data bitcoinotc --emb_size 128 --batch_size 1024 --epochs 30 --learning_rate 0.001 --hidden_dim 64 --patience 30 --gpu 1 --window_size 20 --seq_model lstm
python link_prediction_alongtime.py --data bitcoinotc --emb_size 128 --batch_size 1024 --epochs 30 --learning_rate 0.001 --hidden_dim 64 --patience 30 --gpu 1 --window_size 20 --seq_model gru
python link_prediction_alongtime.py --data bitcoinotc --emb_size 128 --batch_size 1024 --epochs 30 --learning_rate 0.001 --hidden_dim 64 --patience 30 --gpu 1 --window_size 20 --seq_model transformer
```

## Node Classification
```
### Wikipedia
python node_classification.py --batch_size 256 --epochs 30 --emb_size 172 --hidden_dim 128 --learning_rate 0.0001 --gpu 1 --patience 30
### Reddit
python node_classification.py --batch_size 256 --epochs 30 --emb_size 172 --hidden_dim 64 --learning_rate 0.008 --gpu 1 --patience 30 --data reddit
```
