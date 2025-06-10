import argparse


def get_fg_args() -> argparse.Namespace:
    """Configures the arguments for findings generator module.

    Returns:
        argparse.Namespace: Object containing all the configurations.
    """

    parser = argparse.ArgumentParser()

    # Main dataset
    parser.add_argument("--dataset_name", type=str)

    # Files and directories
    parser.add_argument("--images_dir", type=str)
    parser.add_argument("--train_filepath", type=str)
    parser.add_argument("--val_filepath", type=str)
    parser.add_argument("--test_filepath", type=str)
    parser.add_argument("--token2id_filepath", type=str)
    parser.add_argument("--id2token_filepath", type=str)
    parser.add_argument("--checkpoints_dir", type=str)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--logging_dir", type=str)

    # Model configs
    parser.add_argument("--tags_encoder_model_name", type=str)
    parser.add_argument("--impression_encoder_model_name", type=str)
    parser.add_argument("--min_freq", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--impression_seq_len", type=int)
    parser.add_argument("--tag_seq_len", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--prob", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)

    # Optimizer configs
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)

    # Scheduler configs
    parser.add_argument("--T_0", type=int)
    parser.add_argument("--T_mult", type=int)

    # Others
    parser.add_argument("--seed", type=int)

    return parser.parse_args()
