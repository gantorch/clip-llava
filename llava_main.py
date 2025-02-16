import torch
import torchvision
from torchvision.datasets import Caltech101
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T
import wandb

from transformers import LlavaForConditionalGeneration, LlavaProcessor
from pytorch_lightning.loggers import WandbLogger

class Caltech101DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=16, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])

    def prepare_data(self):
        Caltech101(root=self.data_dir, target_type="category", transform=self.transform, download=True)

    def setup(self, stage=None):
        full_dataset = Caltech101(
            root=self.data_dir,
            target_type="category",
            transform=self.transform,
            download=False
        )
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

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

        model_name = "bczhou/tiny-llava-v1-hf"

        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="./offload",
            trust_remote_code=True
        )
        self.processor = LlavaProcessor.from_pretrained(model_name)

        for param in self.llava_model.parameters():
            param.requires_grad = False

        try:
            embedding_dim = self.llava_model.config.vision_config.hidden_size
        except AttributeError:
            raise AttributeError(
                "Could not find hidden_size in self.llava_model.config. "
            )

        self.classifier = torch.nn.Linear(embedding_dim, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, images):
        if getattr(self.processor, "patch_size", None) is None:
            self.processor.patch_size = 16

        inputs = self.processor(images=images, text="", return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            if hasattr(self.llava_model, "get_vision_tower"):
                vision_encoder = self.llava_model.get_vision_tower()
            elif hasattr(self.llava_model, "vision_tower"):
                vision_encoder = self.llava_model.vision_tower
            elif hasattr(self.llava_model, "vision_encoder"):
                vision_encoder = self.llava_model.vision_encoder
            else:
                raise AttributeError("No vision encoder found in the model.")

            vision_outputs = vision_encoder(inputs["pixel_values"])

            if hasattr(vision_outputs, "last_hidden_state"):
                vision_tensor = vision_outputs.last_hidden_state
            elif isinstance(vision_outputs, (tuple, list)):
                vision_tensor = vision_outputs[0]
            else:
                vision_tensor = vision_outputs

            image_embeds = vision_tensor[:, 0, :]

        logits = self.classifier(image_embeds.float())
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
        logger=wandb_logger,
    )

    trainer.fit(model, data_module)

if __name__ == "__main__":
    wandb.login()
    main()
