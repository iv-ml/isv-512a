# Use PYtorch Lightning to train the model using Binary Cross Entropy Loss
import lightning.pytorch as pl
import torch
import torch.nn.functional as F


class SilenceClassifier(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    from silence_classifier.ds import get_data

    train_data, val_data = get_data()
    model = SilenceClassifier()
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_data, val_data)
