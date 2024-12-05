from dataclasses import dataclass
from tqdm import tqdm
import torch as t
from torch import Tensor
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader
import einops
import wandb
import gc

from dataset import MinDataset
from model import create_model


@dataclass
class TrainArgs:
    max_num: int
    seq_len: int
    trainset_size: int
    valset_size: int
    epochs: int
    batch_size: int
    lr: float
    seed: int
    d_model: int
    d_head: int
    n_layers: int
    n_heads: int
    d_mlp: int
    normalization_type: str | None
    use_wandb: bool
    device: str


class Trainer:
    def __init__(self, args: TrainArgs):
        self.args = args
        self.model = create_model(**args.__dict__)
        if args.use_wandb:
            wandb.init(project="min-model")
            wandb.watch(self.model)
        

    def training_step(self, toks: Tensor) -> tuple[t.Tensor, float]:
        logits, target = self._shared_train_validation_step(toks)
        loss = F.cross_entropy(
            logits[:, -1, :],  # Only the last token's logits
            target  # Target is already the last token
        )
        predictions = logits[:, -1, :].argmax(dim=-1)  # Only the last token's logits
        accuracy = (predictions == target).float().mean().item()
        return loss, accuracy

    def validation_step(self, toks: Tensor) -> tuple[float, float]:
        logits, target = self._shared_train_validation_step(toks)
        loss = F.cross_entropy(
            logits[:, -1, :],  # Only the last token's logits
            target  # Target is already the last token
        )
        predictions = logits[:, -1, :].argmax(dim=-1)  # Only the last token's logits
        accuracy = (predictions == target).float().mean().item()
        return loss.item(), accuracy


    def _shared_train_validation_step(self, toks: Tensor) -> tuple[Tensor, Tensor]:
        toks = toks.to(self.args.device)
        inputs = toks[:, :-1]  # All tokens except the last one
        target = toks[:, -1]  # Only the last token
        logits = self.model(inputs)
        return logits, target

    def train_dataloader(self, seed: int):
        trainset = MinDataset(
            size=self.args.trainset_size, max_num=self.args.max_num, length=self.args.seq_len, seed=seed)
        return DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, pin_memory=False, num_workers=0)

    def val_dataloader(self, seed: int):
        valset = MinDataset(
            size=self.args.valset_size, max_num=self.args.max_num, length=self.args.seq_len, seed=seed)
        return DataLoader(valset, batch_size=self.args.batch_size, shuffle=False, pin_memory=False, num_workers=0)

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.model.parameters(), lr=self.args.lr, eps=1e-8)
        return optimizer


def train(args: TrainArgs):
    trainer = Trainer(args)
    optimizer = trainer.configure_optimizers()

    train_dataloader = trainer.train_dataloader(seed=args.seed)
    val_dataloader = trainer.val_dataloader(seed=args.seed + 1)

    best_model = deepcopy(trainer.model)
    best_epoch = None
    best_accuracy = None

    for epoch in range(args.epochs):
        progress_bar = tqdm(total=args.trainset_size // args.batch_size)

        for batch in train_dataloader:
            optimizer.zero_grad()
            loss, accuracy = trainer.training_step(batch)
            loss.backward()
            optimizer.step()
            if args.use_wandb:
                wandb.log({"training_loss": loss, "training_accuracy": accuracy})
            progress_bar.update()
            progress_bar.set_description(f"Epoch {epoch:02}, Train loss = {loss:.4f}, Accuracy = {accuracy:.3f}")

        with t.inference_mode():
            val_loss_list = []
            val_accuracy_list = []
            for batch in val_dataloader:
                val_loss, val_accuracy = trainer.validation_step(batch)
                val_loss_list.append(val_loss)
                val_accuracy_list.append(val_accuracy)
            val_loss = sum(val_loss_list) / len(val_loss_list)
            val_accuracy = sum(val_accuracy_list) / len(val_accuracy_list)
            if args.use_wandb:
                wandb.log({"validation_loss": val_loss, "validation_accuracy": val_accuracy})
            progress_bar.set_description(
                f"Epoch {epoch:02}, Train loss = {loss:.4f}, Accuracy = {accuracy:.3f}, Val loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.3f}"
            )

        if best_accuracy is None or val_accuracy > best_accuracy:
            best_epoch = epoch
            best_accuracy = val_accuracy
            best_model = deepcopy(trainer.model)

        # Clear cache and collect garbage
        t.cuda.empty_cache()
        gc.collect()

    if args.use_wandb:
        wandb.finish()

    print(
        f"Returning best model from epoch {best_epoch}/{args.epochs}, with accuracy {best_accuracy:.3f}"
    )
    return best_model
