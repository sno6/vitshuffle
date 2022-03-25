import torch
from model import VitShuffle, Config
from trainer import Trainer, ImageShuffleDataset

if __name__ == '__main__':
    config = Config(logging=True)

    # Set up wandb for model performance logging.
    if config.logging:
        import wandb
        wandb.init(project="vitshuffle", entity="sno6")
        wandb.config = {
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
        }

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    data = ImageShuffleDataset(config, directory="./data/", device=device)
    model = VitShuffle(config).to(device)

    trainer = Trainer(model, config, data, None)
    trainer.train()
