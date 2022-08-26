
class Config():
    def __init__(self):
        self.npoints = 70
        self.batchsize = 8
        self.batchsize_eval = 1
        self.rpn_epochs = 30
        self.nepochs = 300
        self.n_boxes_min = 10
        self.n_boxes_max = 10
        self.workers = 1
        self.model = ''
        self.out = './checkpoints'
        self.lr = 0.00002
        self.momentum = 0.9
        self.mgpu = False
        self.gpuids = 0
        self.nchannels = 8
        
