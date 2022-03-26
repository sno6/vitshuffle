import torch
from model import VitShuffle, Config
from trainer import Trainer, ImageShuffleDataset

if __name__ == '__main__':
    config = Config(logging=False)

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

    model = VitShuffle(config).to(device)

    train = ImageShuffleDataset(config, directory="./data/train/", device=device, n_images=60000)
    test = ImageShuffleDataset(config, directory="./data/test/", device=device, n_images=10000)
    trainer = Trainer(model, config, train, test)
    trainer.train()
