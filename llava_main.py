import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T
import wandb
from transformers import LlavaProcessor, LlavaForCausalLM

class Caltech101DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=1, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
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
        self.classes = caltech_full.categories
        train_size = int(0.8 * len(caltech_full))
        val_size = len(caltech_full) - train_size
        self.train_dataset, self.val_dataset = random_split(
            caltech_full,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class LLAVAClassifier(pl.LightningModule):
    def __init__(self, llava_model_name="liuhaotian/llava-7b-v0"):
        super().__init__()
        self.processor = LlavaProcessor.from_pretrained(llava_model_name)
        self.model = LlavaForCausalLM.from_pretrained(llava_model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.classes = None

    def setup(self, stage):
        self.classes = self.trainer.datamodule.classes

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        label = labels.item()
        prompt = (
            "You are a helpful assistant for image recognition.\n"
            "User: Here is an image. Can you tell me what the main object in this image is?\n"
            "Assistant: "
        )
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        pred_class = None
        generated_lower = generated_text.lower()
        for c in self.classes:
            if c.lower() in generated_lower:
                pred_class = c
                break

        if pred_class in self.classes:
            pred_idx = self.classes.index(pred_class)
        else:
            pred_idx = -1

        acc = 1.0 if (pred_idx == label) else 0.0
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return acc

    def configure_optimizers(self):
        return []

def main():
    wandb.login()
    wandb_logger = WandbLogger(project="llava-caltech101-zero-shot")
    data_module = Caltech101DataModule(data_dir="./caltech101", batch_size=1)
    model = LLAVAClassifier(llava_model_name="liuhaotian/llava-7b-v0")
    trainer = Trainer(
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger
    )
    trainer.validate(model, datamodule=data_module)

if __name__ == "__main__":
    main()
