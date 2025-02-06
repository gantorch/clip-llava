import torch
import torchvision
from torchvision.datasets import Caltech101
from transformers import CLIPProcessor, CLIPModel
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T
import wandb

class Caltech101DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=32, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),  # ✅ Convert grayscale to RGB
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])

    def setup(self, stage=None):
        caltech_full = torchvision.datasets.Caltech101(
            root=self.data_dir,
            target_type="category",
            transform=self.transform,
            download=True
        )
        train_size = int(0.8 * len(caltech_full))
        val_size = len(caltech_full) - train_size

        self.train_dataset, self.val_dataset = random_split(
            caltech_full,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


class CLIPClassifier(pl.LightningModule):
    def __init__(self, num_classes=102, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        for param in self.clip_model.parameters():
            param.requires_grad = False

        embedding_dim = self.clip_model.config.projection_dim
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(images)
        return self.classifier(image_features)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.lr)


def main():
    wandb_logger = WandbLogger(project="clip-caltech101")

    data_module = Caltech101DataModule(data_dir="./caltech101", batch_size=16, num_workers=0)

    model = CLIPClassifier(num_classes=102, lr=1e-3)

    trainer = Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    wandb.login()
    main()
