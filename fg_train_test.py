import os
import pandas as pd
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.fg import get_fg_args
from configs.seeds import seed_everything
from modules.models import FgModel
from utils.dataloaders import FgDataloaders
from utils.tokenizers import CustomTokenizer
from utils.earlystop import EarlyStopper
from utils.metrics import calculate_metrics
from utils.beam_search import ScoredToken, GeneratedSequence
from transformers import AutoTokenizer
import logging
from einops import rearrange


def get_logger(logging_dir: str) -> logging.Logger:
    """Builds logger.

    Args:
        logging_dir (str): Path to logging directory.

    Returns:
        logging.Logger: Logger.
    """

    if not os.path.isdir(logging_dir):
        os.makedirs(logging_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(f"{logging_dir}/train_test.log", mode="w")
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


class FgTrainer:
    def __init__(
        self,
        model: FgModel,
        loss_fn: nn.CrossEntropyLoss,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: CustomTokenizer,
        device: str,
        logger: logging.Logger,
        epochs: str,
        checkpoints_dir: str,
    ) -> None:
        """Initializes FgTrainer class.

        Args:
            model (FgModel): Whole model.
            loss_fn (nn.CrossEntropyLoss): Loss function.
            optimizer (optim.Optimizer): Optimizer.
            scheduler (optim.lr_scheduler.LRScheduler): Scheduler.
            train_loader (DataLoader): Train loader.
            val_loader (DataLoader): Val loader.
            tokenizer (CustomTokenizer): Tokenizer.
            device (str): Device.
            logger (logging.Logger): Logger.
            epochs (str): Epochs.
            checkpoints_dir (str): Path to checkpoints directory.
        """

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger
        self.epochs = epochs
        self.checkpoints_dir = checkpoints_dir

    def train(self, epoch: int) -> float:
        """Executes training.

        Args:
            epoch (int): Epochs.

        Returns:
            float: Training loss.
        """

        self.model.train()
        running_loss_train: float = 0.0
        iters = len(self.train_loader)

        for batch_idx, batch in tqdm(
            enumerate(self.train_loader), total=len(self.train_loader), leave=True
        ):
            self.optimizer.zero_grad(set_to_none=True)

            images1 = batch["image1"].to(self.device)  # [B, C, H, W]
            images2 = batch["image2"].to(self.device)  # [B, C, H, W]
            impression_input_ids = batch["impression_input_ids"].to(
                self.device
            )  # [B, impression_seq_len]
            impression_attention_masks = batch["impression_attention_mask"].to(
                self.device
            )  # [B, impression_seq_len]
            tags_input_ids = batch["tags_input_ids"].to(self.device)  # [B, tag_seq_len]
            tags_attention_masks = batch["tags_attention_mask"].to(
                self.device
            )  # [B, tag_seq_len]
            input_ids = batch["input_ids"].to(self.device)  # [B, seq_len]
            pad_masks = batch["pad_mask"].to(self.device)  # [B, 1, seq_len]
            labels = batch["label"].to(self.device)  # [B, seq_len]

            outputs = self.model(
                images1,
                images2,
                tags_input_ids,
                tags_attention_masks,
                impression_input_ids,
                impression_attention_masks,
                input_ids,
                pad_masks,
            )  # [B, seq_len, vocab_size]

            loss = self.loss_fn(
                outputs.view(-1, self.tokenizer.get_vocab_size()), labels.view(-1)
            )

            loss.backward()
            self.optimizer.step()
            self.scheduler.step(epoch + batch_idx / iters)
            running_loss_train += loss.item()

        running_loss_train /= len(self.train_loader)
        return running_loss_train

    def validate(self) -> float:
        """Executes validation.

        Returns:
            float: Validation loss.
        """

        with torch.no_grad():
            self.model.eval()
            running_loss_val: float = 0.0

            for batch_idx, batch in tqdm(
                enumerate(self.val_loader), total=len(self.val_loader), leave=True
            ):
                images1 = batch["image1"].to(self.device)  # [B, C, H, W]
                images2 = batch["image2"].to(self.device)  # [B, C, H, W]
                impression_input_ids = batch["impression_input_ids"].to(
                    self.device
                )  # [B, impression_seq_len]
                impression_attention_masks = batch["impression_attention_mask"].to(
                    self.device
                )  # [B, impression_seq_len]
                tags_input_ids = batch["tags_input_ids"].to(
                    self.device
                )  # [B, tag_seq_len]
                tags_attention_masks = batch["tags_attention_mask"].to(
                    self.device
                )  # [B, tag_seq_len]
                input_ids = batch["input_ids"].to(self.device)  # [B, seq_len]
                pad_masks = batch["pad_mask"].to(self.device)  # [B, 1, seq_len]
                labels = batch["label"].to(self.device)  # [B, seq_len]

                outputs = self.model.forward(
                    images1,
                    images2,
                    tags_input_ids,
                    tags_attention_masks,
                    impression_input_ids,
                    impression_attention_masks,
                    input_ids,
                    pad_masks,
                )  # [B, seq_len, vocab_size]

                loss = self.loss_fn(
                    outputs.view(-1, self.tokenizer.get_vocab_size()), labels.view(-1)
                )

                running_loss_val += loss.item()

        running_loss_val /= len(self.val_loader)
        return running_loss_val

    def execute(self) -> None:
        """Executes the training and validation pipeline."""

        if not os.path.isdir(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_loss: float = float("inf")
        early_stopper = EarlyStopper(patience=30, min_delta=0)

        self.model.to(self.device)
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train(epoch)
            val_loss = self.validate()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.logger.info(
                f"Epoch {epoch}/{self.epochs} | Train loss = {train_loss:.4f} | Val loss = {val_loss:.4f} | Last lr = {self.scheduler.get_last_lr()[0]:.4e}"
            )

            if val_loss < best_loss:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                    },
                    f"{self.checkpoints_dir}/best-model.ckpt",
                )
                self.logger.info(
                    f"Saved best model checkpoint at {self.checkpoints_dir}/best-model.ckpt"
                )
                best_loss = val_loss

            self.logger.info(f"Best val loss so far={best_loss:.4f}")

            torch.save(
                {"train_losses": train_losses, "val_losses": val_losses},
                f"{self.checkpoints_dir}/train-val-losses.ckpt",
            )
            self.logger.info(
                f"Saved train-val losses at {self.checkpoints_dir}/train-val-losses.ckpt"
            )

            if early_stopper.early_stop(val_loss):
                self.logger.info(f"Earlystopping at epoch {epoch}")
                break


class FgTester:
    def __init__(
        self,
        model: FgModel,
        test_loader: DataLoader,
        tokenizer: CustomTokenizer,
        device: str,
        logger: logging.Logger,
        checkpoints_dir: str,
        results_dir: str,
        seq_len: int,
        test_filepath: str,
    ) -> None:
        """Initializes FgTester class.

        Args:
            model (FgModel): Whole model.
            test_loader (DataLoader): Test loader.
            tokenizer (CustomTokenizer): Tokenizer.
            device (str): Device.
            logger (logging.Logger): Logger.
            checkpoints_dir (str): Path to checkpoints directory.
            results_dir (str): Path to results directory.
            seq_len (int): Maximum sequence length.
            test_filepath (str): Path to test csv file.
        """

        self.model = model
        self.model.load_state_dict(
            torch.load(f"{checkpoints_dir}/best-model.ckpt")["model"]
        )

        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger
        self.results_dir = results_dir
        self.seq_len = seq_len

        self.df = pd.read_csv(test_filepath, encoding="utf-8")

    def greedy_decode(self) -> None:
        """Applies greedy decoding."""

        self.logger.info("Greedy decoding:")

        identifiers: list[int] = []
        predictions: list[str] = []

        for batch_idx, batch in tqdm(
            enumerate(self.test_loader), total=len(self.test_loader), leave=True
        ):
            sos_id = self.tokenizer.get_id_by_token("<sos>")
            eos_id = self.tokenizer.get_id_by_token("<eos>")
            images1 = batch["image1"].to(self.device)  # [B, C, H, W]
            images2 = batch["image2"].to(self.device)  # [B, C, H, W]
            impression_input_ids = batch["impression_input_ids"].to(
                self.device
            )  # [B, impression_seq_len]
            impression_attention_masks = batch["impression_attention_mask"].to(
                self.device
            )  # [B, impression_seq_len]
            tags_input_ids = batch["tags_input_ids"].to(self.device)  # [B, tag_seq_len]
            tags_attention_masks = batch["tags_attention_mask"].to(
                self.device
            )  # [B, tag_seq_len]

            img1_feats = self.model.vis_feats(images1)  # [B, 1024, 8, 8]
            img2_feats = self.model.vis_feats(images2)  # [B, 1024, 8, 8]
            img_vis_feats = torch.cat(
                [img1_feats, img2_feats], dim=1
            )  # [B, 2048, 8, 8]
            img_vis_feats = rearrange(
                img_vis_feats, "b c h w -> b (h w) c"
            )  # [B, 64, 2048]
            img_vis_feats = self.model.vis_linear(img_vis_feats)  # [B, 64, d_model]

            imp_embs = self.model.imp_embs(
                impression_input_ids, impression_attention_masks
            )  # [B, impression_seq_len, 768]
            imp_embs = self.model.imp_linear(
                imp_embs
            )  # [B, impression_seq_len, d_model]

            tag_embs = self.model.tag_embs(
                tags_input_ids, tags_attention_masks
            )  # [B, tag_seq_len, 768]
            tag_embs = self.model.tag_linear(tag_embs)  # [B, tag_seq_len, d_model]

            memory = torch.cat(
                [img_vis_feats, tag_embs, imp_embs], dim=1
            )  # [B, total_seq_len, d_model]

            decoder_input = torch.empty(
                (1, 1), dtype=torch.int64, device=self.device
            ).fill_(sos_id)  # [B, seq_len]

            while True:
                if decoder_input.size(1) == self.seq_len:
                    break

                causal_mask = FgModel.create_causal_mask(
                    decoder_input
                )  # [B, seq_len, seq_len]
                cross_mask = FgModel.create_encoder_mask_for_cross_attention(
                    decoder_input.size()[1],
                    img_vis_feats.size()[1],
                    tags_attention_masks,
                    impression_attention_masks,
                )
                rep_emb = self.model.rep_emb(decoder_input)  # [B, seq_len, d_model]
                dec_out = self.model.decoder_layers(
                    rep_emb, memory, causal_mask, cross_mask
                )  # [B, seq_len, d_model]
                last_seq_out = dec_out[:, -1].unsqueeze(0)  # [1, 1, d_model]
                logits = self.model.logits(
                    self.model.final_rms_norm(last_seq_out)
                ).squeeze(0)  # [1, vocab_size]

                next_word_id = torch.argmax(logits, dim=-1).item()
                decoder_input = torch.cat(
                    [
                        decoder_input,
                        torch.empty(
                            (1, 1), dtype=torch.int64, device=self.device
                        ).fill_(next_word_id),
                    ],
                    dim=1,
                )  # [B, seq_len]
                if next_word_id == eos_id:
                    break

            decoded_seq = self.tokenizer.decode_by_ids(
                decoder_input.squeeze(0).cpu().numpy().tolist()
            )

            identifiers.append(batch["identifier"][0].item())
            predictions.append(decoded_seq)

        temp_df = pd.DataFrame(
            {
                "identifier": identifiers,
                "prediction_fg": predictions,
            }
        )
        merged_df = pd.merge(self.df, temp_df, on="identifier")
        merged_df.to_csv(f"{self.results_dir}/greedy-predictions.csv")
        self.logger.info(
            f"Saved greedy decoded predictions at {self.results_dir}/greedy-predictions.csv"
        )

    def beam_search_decode(self) -> None:
        """Applies beam search decoding."""

        self.logger.info("Beam search decoding:")

        identifiers: list[int] = []
        predictions: list[str] = []

        for batch_idx, batch in tqdm(
            enumerate(self.test_loader), total=len(self.test_loader), leave=True
        ):
            sos_id = self.tokenizer.get_id_by_token("<sos>")
            eos_id = self.tokenizer.get_id_by_token("<eos>")
            images1 = batch["image1"].to(self.device)  # [B, C, H, W]
            images2 = batch["image2"].to(self.device)  # [B, C, H, W]
            impression_input_ids = batch["impression_input_ids"].to(
                self.device
            )  # [B, impression_seq_len]
            impression_attention_masks = batch["impression_attention_mask"].to(
                self.device
            )  # [B, impression_seq_len]
            tags_input_ids = batch["tags_input_ids"].to(self.device)  # [B, tag_seq_len]
            tags_attention_masks = batch["tags_attention_mask"].to(
                self.device
            )  # [B, tag_seq_len]

            img1_feats = self.model.vis_feats(images1)  # [B, 1024, 8, 8]
            img2_feats = self.model.vis_feats(images2)  # [B, 1024, 8, 8]
            img_vis_feats = torch.cat(
                [img1_feats, img2_feats], dim=1
            )  # [B, 2048, 8, 8]
            img_vis_feats = rearrange(
                img_vis_feats, "b c h w -> b (h w) c"
            )  # [B, 64, 2048]
            img_vis_feats = self.model.vis_linear(img_vis_feats)  # [B, 64, d_model]

            imp_embs = self.model.imp_embs(
                impression_input_ids, impression_attention_masks
            )  # [B, impression_seq_len, 768]
            imp_embs = self.model.imp_linear(
                imp_embs
            )  # [B, impression_seq_len, d_model]

            tag_embs = self.model.tag_embs(
                tags_input_ids, tags_attention_masks
            )  # [B, tag_seq_len, 768]
            tag_embs = self.model.tag_linear(tag_embs)  # [B, tag_seq_len, d_model]

            memory = torch.cat(
                [img_vis_feats, tag_embs, imp_embs], dim=1
            )  # [B, total_seq_len, d_model]

            beam_width: int = 5
            candidate_seq = [GeneratedSequence(self.tokenizer, sos_id, eos_id, 0.0)]
            for i in range(self.seq_len):
                next_step_candidates = []
                for candidate in candidate_seq:
                    if not candidate.has_ended():
                        decoder_input = torch.tensor(
                            candidate.ids(), device=self.device, dtype=torch.int64
                        ).unsqueeze(0)  # [B, seq_len]
                        causal_mask = FgModel.create_causal_mask(
                            decoder_input
                        )  # [B, seq_len, seq_len]
                        cross_mask = FgModel.create_encoder_mask_for_cross_attention(
                            decoder_input.size()[1],
                            img_vis_feats.size()[1],
                            tags_attention_masks,
                            impression_attention_masks,
                        )
                        rep_emb = self.model.rep_emb(
                            decoder_input
                        )  # [B, seq_len, d_model]
                        dec_out = self.model.decoder_layers(
                            rep_emb, memory, causal_mask, cross_mask
                        )  # [B, seq_len, d_model]
                        last_seq_out = dec_out[:, -1].unsqueeze(0)  # [1, 1, d_model]
                        logits = self.model.logits(
                            self.model.final_rms_norm(last_seq_out)
                        ).squeeze(0)  # [1, vocab_size]
                        probs = torch.softmax(logits, dim=-1)
                        top_probs, top_ids = probs.topk(beam_width)

                        for i in range(beam_width):
                            next_token_id = top_ids[:, i].item()
                            next_score = torch.log(top_probs[:, i]).item()
                            new_seq = deepcopy(candidate)
                            new_seq.append(ScoredToken(next_token_id, next_score))
                            next_step_candidates.append(new_seq)
                    else:
                        next_step_candidates.append(candidate)

                next_step_candidates.sort()
                candidate_seq = list(reversed(next_step_candidates))[:beam_width]
                if all(seq.has_ended() for seq in candidate_seq):
                    break

            decoded_seq = candidate_seq[0].decode_tokens()

            identifiers.append(batch["identifier"][0].item())
            predictions.append(decoded_seq)

        temp_df = pd.DataFrame(
            {
                "identifier": identifiers,
                "prediction_fg": predictions,
            }
        )
        merged_df = pd.merge(self.df, temp_df, on="identifier")
        merged_df.to_csv(f"{self.results_dir}/beam-search-predictions.csv")
        self.logger.info(
            f"Saved beam search decoded predictions at {self.results_dir}/beam-search-predictions.csv"
        )

    def execute(self) -> None:
        """Function to execute testing pipeline. Applies greedy decoding and beam search decoding."""

        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)

        self.model.to(self.device)

        with torch.no_grad():
            self.model.eval()

            self.greedy_decode()
            self.logger.info("Results of greedy decoding:")
            df = pd.read_csv(
                f"{self.results_dir}/greedy-predictions.csv", encoding="utf-8"
            )
            preds: list[str] = []
            refs: list[str] = []
            for pred, ref in zip(
                df["prediction_fg"].values.tolist(), df["findings"].values.tolist()
            ):
                preds.append(pred)
                refs.append(ref)
            calculate_metrics(preds, refs, self.logger)

            self.beam_search_decode()
            self.logger.info("Results of beam search decoding:")
            df = pd.read_csv(
                f"{self.results_dir}/beam-search-predictions.csv", encoding="utf-8"
            )
            preds: list[str] = []
            refs: list[str] = []
            for pred, ref in zip(
                df["prediction_fg"].values.tolist(), df["findings"].values.tolist()
            ):
                preds.append(pred)
                refs.append(ref)
            calculate_metrics(preds, refs, self.logger)


def main() -> None:
    """Executes the main flow."""

    args = get_fg_args()

    logger = get_logger(args.logging_dir)

    logger.info(f"Experiment args: {args}")

    seed_everything(args.seed)

    tokenizer = CustomTokenizer(
        args.train_filepath,
        args.min_freq,
        args.token2id_filepath,
        args.id2token_filepath,
        args.seq_len,
        "findings",
    )

    tag_tokenizer = AutoTokenizer.from_pretrained(args.tags_encoder_model_name)
    impression_tokenizer = AutoTokenizer.from_pretrained(
        args.impression_encoder_model_name
    )

    model = FgModel(args, tokenizer.get_vocab_size())
    logger.info(f"Total params={sum(p.numel() for p in model.parameters())}")
    logger.info(
        f"Total trainable params={sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, args.T_0, args.T_mult
    )

    train_loader = FgDataloaders.get_train_loader(
        args.dataset_name,
        args.images_dir,
        args.train_filepath,
        args.batch_size,
        tokenizer,
        tag_tokenizer,
        args.tag_seq_len,
        impression_tokenizer,
        args.impression_seq_len,
    )

    val_loader = FgDataloaders.get_val_loader(
        args.dataset_name,
        args.images_dir,
        args.val_filepath,
        args.batch_size,
        tokenizer,
        tag_tokenizer,
        args.tag_seq_len,
        impression_tokenizer,
        args.impression_seq_len,
    )

    test_loader = FgDataloaders.get_test_loader(
        args.dataset_name,
        args.images_dir,
        args.test_filepath,
        tokenizer,
        tag_tokenizer,
        args.tag_seq_len,
        impression_tokenizer,
        args.impression_seq_len,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.get_id_by_token("<pad>"))

    if torch.cuda.is_available():
        device: str = "cuda"
        logger.info(f"Device: {torch.cuda.get_device_properties('cuda')}")
    else:
        device: str = "cpu"
        logger.info("Device: CPU")

    trainer = FgTrainer(
        model,
        loss_fn,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        tokenizer,
        device,
        logger,
        args.epochs,
        args.checkpoints_dir,
    )
    trainer.execute()

    tester = FgTester(
        model,
        test_loader,
        tokenizer,
        device,
        logger,
        args.checkpoints_dir,
        args.results_dir,
        args.seq_len,
        args.test_filepath,
    )
    tester.execute()


if __name__ == "__main__":
    main()
