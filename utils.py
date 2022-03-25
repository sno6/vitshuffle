def blockshaped(arr, n_row, n_col):
    h, w = arr.shape
    return (arr.reshape(h // n_row, n_row, -1, n_col)
            .swapaxes(1, 2)
            .reshape(-1, n_row, n_col))
