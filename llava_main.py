import torch
import torchvision
from torchvision.datasets import Caltech101
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T
import wandb

# ✅ Import Correct LLaVA Model & Processor
from transformers import LlavaForConditionalGeneration, LlavaProcessor

class Caltech101DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=32, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])

    def setup(self, stage=None):
        caltech_full = Caltech101(
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class LLAVAClassifier(pl.LightningModule):
    def __init__(self, num_classes=102, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # ✅ Use the correct LLaVA model
        model_name = "liuhaotian/llava-13b-v1.5"

        # ✅ Use `LlavaForConditionalGeneration` instead of AutoModelForVision2Seq
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="./offload",  # Fix memory issues
            trust_remote_code=True
        )

        # ✅ Use `LlavaProcessor`
        self.processor = LlavaProcessor.from_pretrained(model_name)

        # Freeze all LLaVA parameters (only train classifier)
        for param in self.llava_model.parameters():
            param.requires_grad = False

        # Get embedding dimension
        embedding_dim = self.llava_model.config.hidden_size  # Typically 4096 or 5120 for LLaVA

        # Define trainable classifier
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, images):
        """
        Pass images through the LLaVA vision encoder and get embeddings.
        Then apply the trainable classification head.
        """
        with torch.no_grad():
            # ✅ Process images using LLaVA processor
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

            # ✅ Get image embeddings from LLaVA’s vision tower
            vision_outputs = self.llava_model.get_vision_tower()(inputs["pixel_values"])

            # ✅ Extract [CLS] token embedding or pool across all tokens
            image_embeds = vision_outputs[:, 0, :]  # Shape [B, hidden_dim]

        logits = self.classifier(image_embeds)  # Shape [B, num_classes]
        return logits

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
    wandb_logger = WandbLogger(project="clip-llava-caltech101")

    data_module = Caltech101DataModule(data_dir="./caltech101", batch_size=16, num_workers=0)
    model = LLAVAClassifier(num_classes=102, lr=1e-3)

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
