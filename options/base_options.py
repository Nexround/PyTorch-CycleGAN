import argparse
import sys


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.n_cpu = 0 if sys.platform == 'win32' else 8


    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
        parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
        parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
        parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
        parser.add_argument('--A', type=str, help='A directory')
        parser.add_argument('--B', type=str, help='B directory')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
        parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
        parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
        parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
        parser.add_argument('--cuda', default=True, action='store_true', help='use GPU computation')
        parser.add_argument('--n_cpu', type=int, default=self.n_cpu, help='number of cpu threads to use during batch generation')
        parser.add_argument('--use_wandb', default=False, action='store_true', help='use wandb for logging')
        self.initialized = True
        return parser
    
    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        self.print_options(opt)

        self.opt = opt
        return self.opt


