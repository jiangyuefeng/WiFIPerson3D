import warnings


class DefaultConfig(object):
    model = 'WiFiModel'

    train_data = '/home/public/b509/code/g19/jyf/mypro/PersonWiFi3/dataset'
    test_data = '/run/user/1000/gvfs/smb-share:server=509lw.local,share=code/g19/jyf/mypro/PersonWiFi3/dataset'
    load_model_path = ''
    batch_size = 32
    use_gpu = True
    num_workers = 4
    print_freq = 20

    debug_file = 'tmp/debug'
    result_file = 'result.csv'

    max_epoch = 20
    lr = 0.000001
    lr_decay = 0.95
    weight_decay = 1e-4


def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
# opt.parse = parse
