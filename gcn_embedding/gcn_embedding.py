import pickle
import torch
from src.utils import create_graphs, create_embeddings
import argparse

def main():
    parser = argparse.ArgumentParser(description='Create embeddings and save to disk.')
    parser.add_argument('--data_dir', type=str, default='gcn/data/emb', help='Directory to save the data.')
    parser.add_argument('--output-file', type=str, default='gcn/data/emb/embeddings.pkl', help='File to save the embeddings')
    parser.add_argument('--p_value', type=float, default=0.05, help='P-value threshold for creating embeddings.')
    parser.add_argument('--save', type=bool, default=True, help='Flag to save embeddings.')
    parser.add_argument('--num_epochs', type=int, default=5000, help='Number of epochs for training.')
    parser.add_argument('--in_feats', type=int, default=128, help='Number of input features.')
    parser.add_argument('--out_feats', type=int, default=128, help='Number of output features.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model.')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of heads for GAT model.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--print-embeddings', action='store_true', help='Print the embeddings dictionary')
    parser.add_argument('--feat_drop', type=float, default=0.0, help='Feature dropout rate')
    parser.add_argument('--attn-drop', type=float, default=0.0, help='Attention dropout rate')

    args = parser.parse_args()

    # Main script to create embeddings and save to disk
    ## create embeddings in data/ebm/raw first
    create_graphs(
        ## p_value=args.p_value, 
        save=args.save, 
        data_dir=args.data_dir
    )
    
    hyperparameters = {
        'num_epochs': args.num_epochs,
        'in_feats': args.in_feats,
        'out_feats': args.out_feats,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,  
        'feat_drop': args.feat_drop,
        'attn_drop': args.attn_drop,
        'batch_size': args.batch_size,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': args.lr,
    }
    
    embedding_dict = create_embeddings(
        data_dir=args.data_dir, 
        load_model=False, 
        hyperparams=hyperparameters
    )
    
    
    # Print the embeddings dictionary if required
    if args.print_embeddings:
        print(embedding_dict)

    # Save embeddings to file
    with open(args.output_file, 'wb') as f:
        pickle.dump(embedding_dict, f)
    print(f"Embeddings saved to {args.output_file}")
    
if __name__ == '__main__':

    main()


## python gcn/gcn_embedding.py --out_feats 128 --num_layers 3 --num_heads 2 --batch_size 1 --lr 0.0001 --num_epochs 2002
## python gcn_src_mirna_db/gcn_embedding.py --out_feats 128 --num_layers 2 --num_heads 2 --batch_size 1 --lr 0.0001 --num_epochs 10
## python gcn_src_mirna_db/gcn_embedding.py --out_feats 256 --num_layers 2 --num_heads 2 --batch_size 1 --lr 0.00001 --num_epochs 20