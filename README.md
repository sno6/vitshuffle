## A simple visual transformer pre-training idea.

Given an input image:

![](https://github.com/sno6/vitshuffle/blob/main/examples/normal.png)

We simply break the image into n x n blocks, and shuffle:

![](https://github.com/sno6/vitshuffle/blob/main/examples/shuffled.png)

The goal of the network is then to predict, for each block in the input sequence, where it belongs in the original image.

TODO:

- Continue testing against large datasets. The fact that pre-training needs a lot of data, and GPUs are hard to acquire, makes this difficult.
- I think relative segments is more important than absolute positioning, and should be factored into loss.
- The network can not currently learn translation (of image), a relative segment loss should fix this.
- Update README with instructions for running this thing.