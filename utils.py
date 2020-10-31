import numpy as np


class DataGenerator:
    def __init__(self, inputs, shuffle=True, batch_size=32):
        assert len(inputs) > 0
        self.inputs = inputs
        self.idx = np.arange(len(inputs[0]))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def data_length(self):
        return len(self.idx)

    def __len__(self):
        n = self.data_length()
        len_ = n // self.batch_size
        return len_ if n % self.batch_size == 0 else len_ + 1

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        index = self.idx[start:end]
        data = []
        for x in self.inputs:
            data.append(x[index])
        return data

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


def lr_decay(total_epoch, init_lr, split_val):
    lr_map = [init_lr] * total_epoch
    if len(split_val) > 0:
        assert split_val[0][0] > 1
        assert split_val[-1][0] <= total_epoch
        current_split_index = 0
        current_lr = init_lr
        next_epoch, next_lr = split_val[current_split_index]
        for i in range(total_epoch):
            if i < next_epoch - 1:
                lr_map[i] = current_lr
            else:
                current_lr = next_lr
                lr_map[i] = current_lr
                current_split_index += 1
                if current_split_index >= len(split_val):
                    next_epoch = total_epoch + 1
                else:
                    next_epoch, next_lr = split_val[current_split_index]

    def lr_schedule_fn(epoch, lr):
        return lr_map[epoch]

    return lr_schedule_fn
