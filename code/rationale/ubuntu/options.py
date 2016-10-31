
import sys
import argparse

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--corpus",
            type = str
        )
    argparser.add_argument("--train",
            type = str,
            default = ""
        )
    argparser.add_argument("--test",
            type = str,
            default = ""
        )
    argparser.add_argument("--dev",
            type = str,
            default = ""
        )
    argparser.add_argument("--dump_rationale",
            type = str,
            default = ""
        )
    argparser.add_argument("--embeddings",
            type = str,
            default = ""
        )
    argparser.add_argument("--hidden_dim", "-d",
            type = int,
            default = 400
        )
    argparser.add_argument("--hidden_dim2", "-d2",
            type = int,
            default = 100
        )
    argparser.add_argument("--learning",
            type = str,
            default = "adam"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = 0.0005
        )
    argparser.add_argument("--l2_reg",
            type = float,
            default = 1e-7
        )
    argparser.add_argument("--activation", "-act",
            type = str,
            default = "tanh"
        )
    argparser.add_argument("--batch_size",
            type = int,
            default = 64
        )
    argparser.add_argument("--depth",
            type = int,
            default = 1
        )
    argparser.add_argument("--dropout",
            type = float,
            default = 0.1
        )
    argparser.add_argument("--max_epoch",
            type = int,
            default = 50
        )
    argparser.add_argument("--max_seq_len",
            type = int,
            default = 100
        )
    argparser.add_argument("--normalize",
            type = int,
            default = 0
        )
    argparser.add_argument("--reweight",
            type = int,
            default = 1
        )
    argparser.add_argument("--order",
            type = int,
            default = 2
        )
    argparser.add_argument("--layer",
            type = str,
            default = "rcnn"
        )
    argparser.add_argument("--mode",
            type = int,
            default = 1
        )
    argparser.add_argument("--outgate",
            type = int,
            default = 0
        )
    argparser.add_argument("--load_pretrain",
            type = str,
            default = ""
        )
    argparser.add_argument("--average",
            type = int,
            default = 0
        )
    argparser.add_argument("--save_model",
            type = str,
            default = ""
        )
    argparser.add_argument("--sparsity",
            type = float,
            default = 0.01
        )
    argparser.add_argument("--coherent",
            type = float,
            default = 1.0
        )
    argparser.add_argument("--alpha",
            type = float,
            default = 0.5
        )
    argparser.add_argument("--beta",
            type = float,
            default = 0.1
        )
    argparser.add_argument("--joint",
            type = int,
            default = 1
        )
    argparser.add_argument("--merge",
            type = int,
            default = 1
        )
    args = argparser.parse_args()
    print args
    print ""
    return args
