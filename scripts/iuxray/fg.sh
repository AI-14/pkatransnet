python fg_train_test.py \
--dataset_name iuxray \
--images_dir datasets/iuxray/images \
--train_filepath datasets/iuxray/train.csv \
--val_filepath datasets/iuxray/val.csv \
--test_filepath datasets/iuxray/test.csv \
--token2id_filepath datasets/iuxray/fg-token2id.json \
--id2token_filepath datasets/iuxray/fg-id2token.json \
--checkpoints_dir fg-ckpts \
--logging_dir fg-logs \
--results_dir fg-res \
--tags_encoder_model_name emilyalsentzer/Bio_ClinicalBERT \
--impression_encoder_model_name emilyalsentzer/Bio_ClinicalBERT \
--min_freq 3 \
--seq_len 60 \
--impression_seq_len 40 \
--tag_seq_len 100 \
--num_layers 3 \
--num_heads 8 \
--d_model 512 \
--d_ff 2048 \
--prob 0.1 \
--batch_size 8 \
--epochs 150 \
--lr 5e-5 \
--weight_decay 5e-4 \
--T_0 10 \
--T_mult 1 \
--seed 1234