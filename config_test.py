import warnings


class DefaultConfig(object):
    model = 'WiFiModel'

    train_data = "/home/public/b509/code/dataset"
    test_data = '/home/public/b509/code/dataset'
    load_model_path = 'checkpoints/WiFiModel_0309_15_56_53.pth'
    batch_size = 32
    num_workers = 4
    print_freq = 20
    use_gpu = True

    debug_file = 'tmp/debug'
    result_file = 'result.csv'

    max_epoch = 20
    lr = 0.001
    lr_decay = 0.95
    # weight_decay = 1e-4


def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
# opt.parse = parse
