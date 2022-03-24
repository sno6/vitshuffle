import os
import torch
import wandb
import random

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from PIL import Image, ImageOps
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import to_tensor


class ImageShuffleDataset(Dataset):
    def __init__(self, config, directory, device):
        super().__init__()

        self.config = config
        self.data_directory = directory
        self.device = device

    def __getitem__(self, idx):
        n_patches = int((self.config.image_size / self.config.patch_size) ** 2)

        image = Image.open(os.path.join(self.data_directory, f'{idx}.png'))
        image = ImageOps.grayscale(image)
        image = to_tensor(image)


        # image.size() = (B, W, H) = (1, 28, 28)
        # We want image_size**2 / patch_size**2 patches of patch_size**2.
        b, w, h = image.size()
        assert w == h and w % self.config.patch_size == 0

        blocks = self.blockshaped(image[0], self.config.patch_size, self.config.patch_size)
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

    def blockshaped(self, arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
        assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
        return (arr.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

    def __len__(self):
        return 10000


class Trainer:
    def __init__(self, model, config, train_dataset, test_dataset=None):
        self.model = model
        self.config = config
        self.train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)

        self.test_data = None
        if test_dataset:
            self.test_data = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

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

        av_loss = 0
        for i, (batch_x, batch_y) in enumerate(self.test_data):
            _, loss = self.model(batch_x, batch_y)
            av_loss += loss

        av_loss /= len(self.test_data)
        print(f"Average test loss: {av_loss}")

        if self.config.logging:
            wandb.log({"test_loss": av_loss})