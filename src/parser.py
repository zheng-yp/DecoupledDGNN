import argparse

def parameter_parser():

	 parser = argparse.ArgumentParser(description = "Tweet Classification")

	 parser.add_argument("--epochs",
								dest = "epochs",
								type = int,
								default = 50,
						 help = "Number of gradient descent iterations. Default is 200.")

	 parser.add_argument("--learning_rate",
								dest = "learning_rate",
								type = float,
								default = 0.01,
						 help = "Gradient descent learning rate. Default is 0.01.")

	 parser.add_argument("--hidden_dim",
								dest = "hidden_dim",
								type = int,
								default = 128,
						 help = "Number of neurons by hidden layer. Default is 128.")
						 
	 parser.add_argument("--lstm_layers",
								dest = "lstm_layers",
								type = int,
								default = 2,
					 help = "Number of LSTM layers")
					 
	 parser.add_argument("--batch_size",
									dest = "batch_size",
									type = int,
									default = 64,
							 help = "Batch size")

	 parser.add_argument("--dropout",
								dest = "dropout",
								type = float,
								default = 0.1,
						 help = "the number of dropout.")
						 
	 parser.add_argument("--max_len",
								dest = "max_len",
								type = int,
								default = 20,
						 help = "Maximum sequence length per tweet")

	 # parser.add_argument("--max_words",
		# 						dest = "max_words",
		# 						type = float,
		# 						default = 256, #344 ##for wikipedia  #7230,
		# 				 help = "Maximum number of words in the dictionary")

	 parser.add_argument("--data",
								default = 'wikipedia',
						help = "dataset name")

	 parser.add_argument("--seq_model", default = 'lstm', help = "sequence model type")
	 parser.add_argument("--window_size",
								dest = "window_size",
								type = int,
								default = 20,
						help = "window size")

	 parser.add_argument("--emb_size",
								dest = "emb_size",
								type = int,
								default = 256,
						help = "embedding size")

	 parser.add_argument("--seed",
								dest = "seed",
								type = int,
								default = 1024,
						help = "random seed")

	 parser.add_argument("--gpu",
								dest = "gpu",
								type = int,
								default = 0,
						help = "ID number of gpu")

	 parser.add_argument("--patience",
								dest = "patience",
								type = int,
								default = 3,
						help = "max number of bad count")

	 parser.add_argument("--checkpt_path",
								default = './check_point',
						help = "check ponit path")
	 parser.add_argument('--inductive', action='store_true', help='Whether to use inductive setting')
	 parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle training data')
	 parser.add_argument('--use_neg', action='store_true', help='Whether to use neg feat')

	 parser.add_argument("--alpha",
								dest = "alpha",
								type = float,
								default = 0.98,
						 help = "parameter for FocalLoss - alpha")
	 parser.add_argument("--gamma",
								dest = "gamma",
								type = int,
								default = 4,
						 help = "parameter for FocalLoss - gamma")
	 return parser.parse_args()
