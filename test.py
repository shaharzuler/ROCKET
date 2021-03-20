from model import RocketNet

if __name__ == '__main__':
    model = RocketNet(
        x_dim=8,
        n_classes=2,
        kernel_count=10000,
        max_sequence_len=256,
        kernel_lengths=[7, 9, 11])

    print(model)


