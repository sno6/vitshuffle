import os
import torch
import wandb
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
from utils import blockshaped


class ImageShuffleDataset(Dataset):
    def __init__(self, config, directory, device, n_images):
        super().__init__()

        self.n_images = n_images
        self.config = config
        self.data_directory = directory
        self.device = device

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_directory, f'{idx}.png'))
        image = ImageOps.grayscale(image)
        image = to_tensor(image).squeeze(0)

        w, h = image.size()
        assert w == h and w % self.config.patch_size == 0

        n_patches = int((self.config.image_size / self.config.patch_size) ** 2)
        blocks = blockshaped(image, self.config.patch_size, self.config.patch_size)
        blocks = blocks.view(n_patches, self.config.patch_size**2).contiguous()

        shuffled_positions = [i for i in range(0, blocks.size(0))]
        random.shuffle(shuffled_positions)

        # Manually rejig the blocks to match the shuffled positions.
        shuffled_blocks = []
        for i in shuffled_positions:
            shuffled_blocks.append(blocks[i])

        return (
            torch.stack(shuffled_blocks, 0).to(self.device),
            torch.tensor(shuffled_positions, dtype=torch.long).to(self.device),
        )

    def plot_blocks(self, blocks):
        sz = int(self.config.image_size / self.config.patch_size)
        _, axs = plt.subplots(sz, sz, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(blocks, axs):
            ax.imshow(img)
        plt.show()

    def __len__(self):
        return self.n_images


class Trainer:
    def __init__(self, model, config, train_dataset, test_dataset=None):
        self.model = model
        self.config = config
        self.train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)

        self.test_data = None
        if test_dataset:
            self.test_data = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    def train(self):
        self.model.train()

        for epoch in range(self.config.epochs):
            for i, (batch_x, batch_y) in enumerate(self.train_data):
                with torch.set_grad_enabled(True):
                    y_hat, loss = self.model(batch_x, batch_y)
                    if i % self.config.print_loss_every_iter == 0:
                        print("Loss: ", loss)

                if self.config.logging:
                    wandb.log({"loss": loss})

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.test_data and epoch > 0 and epoch % self.config.test_every_n_epochs == 0:
                self.test()

            # Save a checkpoint at each epoch.
            if epoch % self.config.save_chkpt_every_n_epochs == 0:
                torch.save(self.model.state_dict(), f"chkpt-{epoch}.pt")

    def test(self):
        self.model.eval()

        if not self.test_data or len(self.test_data) == 0:
            return

        with torch.no_grad():
            av_loss = 0
            for i, (batch_x, batch_y) in enumerate(self.test_data):
                _, loss = self.model(batch_x, batch_y)
                av_loss += loss

            av_loss /= len(self.test_data)
            print(f"Average test loss: {av_loss}")

            if self.config.logging:
                wandb.log({"test_loss": av_loss})