class Param(object):
    def __init__(self):
        # Path
        self.root = 'Z:/2nd_paper'
        self.dataset_path = f'{self.root}/dataset/ND/full_image/CycleGAN/1-fold'

        self.output_ckp = f'{self.root}/backup/PAD/FastFakeDetector/1-fold/try3/ckp'
        self.output_log = f'{self.root}/backup/PAD/FastFakeDetector/1-fold/try3/log'

        # Check Point
        self.load_ckp_epoch = 0
        self.do_ckp_load = False

        # Data
        self.data_folder = ['A', 'B']
        self.cls_folder = ['fake', 'live']
        self.input_size = 224

        # Model HyperParameter
        self.depth = 32
        self.dim = 768
        self.kernel_size = 9
        self.patch_size = 7
        self.n_classes = 2

        # Train or Test
        self.full_epoch = 300
        self.lr = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 8
        self.device = 'cuda'

        self.do_display = True

        # Handler
        # run_type [0 : train], [1 : test score], [2 : testing output image]
        self.run_type = 0