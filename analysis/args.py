
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow as tf 

import os
import io
import IPython.display
import numpy as np
import PIL.Image
from scipy.stats import truncnorm
# import tensorflow_hub as hub
import argparse
import json
from file_util import *


def add_boolean_option (parser, arg_name, default=False, false_name=None, help=None, help_false=None):
    """
    Create boolean option for argparse.  With default Name and default value and false name if value is False for disambiguation.
    @param parser: Argparser
    @param arg_name:  default argument name
    @param default: default value
    @param false_name:  Name for False boolean value
    @param help:
    @param help_false:
    @return:
    """
    if false_name is None:
        false_name = 'no_' + arg_name
    parser.add_argument('--' + arg_name, dest=arg_name, action='store_true', help=help)
    parser.add_argument('--' + false_name, dest=arg_name, action='store_false', help=help_false)
    parser.set_defaults(**{arg_name : default})


def get_parser():
    """
    Command line options parser.
    If run pre-specified model run config file of model in tests/
    @return: arguments
    """
    parser = argparse.ArgumentParser(description='RMT GAN')
    parser.add_argument('--num_imgs_per_class', type=int, help="num_imgs_per_class")
    parser.add_argument('--num_gram_imgs_per_class', type=int, help="num_gram_imgs_per_class")
    parser.add_argument('--z_dim', type=int, default=128, help='noise z dimension')
    parser.add_argument('--batch_size', type=int, default=10, help='bath size for generating images')
    parser.add_argument('--classes', nargs='+', type=int, default=[16],  help='classes')
    # parser.add_argument('-d', '--gmm', type=json.loads)
    parser.add_argument('-z', '--z_sampler_args', type=json.loads)
    parser.add_argument('-g', '--g_sampler_args', type=json.loads)
    parser.add_argument('--truncation', type=float, default=1.0, help='truncation parameter in GAN')
    # simple gan parameters
    parser.add_argument('--keras_gan', type=json.loads)
    parser.add_argument('--run', type=str)
    parser.add_argument('--exp_n', type=int, default=0, help='experiment run times')
    
    parser.add_argument('--represent_way', type=str)
    parser.add_argument('--rep_model_name', type=str)
    parser.add_argument('--gan_name', type=str)
    parser.add_argument('--analysis_type', type=str)


    add_boolean_option(parser, 'draw_distribution', default=False, help='draw distribution or not')

    add_boolean_option(parser, 'generate_gmm_data', default=False, help='generate gmm data or not')

    add_boolean_option(parser, 'gramm_analysis', default=False, help='gram analysis or not')

    add_boolean_option(parser, 'singular_analysis', default=False, help='singular analysis or not')
    
    add_boolean_option(parser, 'save_npy', default=False, help='if or not save npy file after cnn')

    add_boolean_option(parser, 'always_new', default=True, help='if or not always generate and represent image')

    add_boolean_option(parser, 'pretrained', default=True, help='if or not pretrained for the CNN')


    # parser.add_argument('--config', type=open, action=LoadFromJSONFile, \
    #                     help='A JSON file storing experiment configuration. This can be found in the tests directory')
    # parser.add_argument('--model_path', type=str, help='Directory for model hub')
    # parser.add_argument('--tfhub_model_load_format', type=str, default='COMPRESSED', help='TFHUB_MODEL_LOAD_FORMAT')
    # parser.add_argument('--num_samples', type=int, help="@param {type:'slider', min:1, max:20, step:1}")
    # num_samples = 10 #
    # noise_seed = 0 #@param {type:"slider", min:0, max:100, step:1}
    # parser.add_argument('--noise_seed', type=int, help="@param {type:'slider', min:0, max:100, step:1}")
    # parser.add_argument('--truncation', type=float, help="@param {type:'slider', min:0.02, max:1, step:0.02}")
    # truncation = 0.4 #@param {type:"slider", min:0.02, max:1, step:0.02}
    # parser.add_argument('--category', type=str, help='Category for image')
    # parser.add_argument('--sample_method', type=str, help='sample method for input noise')
    # category = "933) cheeseburger" 
    # parser.add_argument('--datapath', type=str, \
    #                     help='Directory containing trained_models. Each pkl file contains matrices W2ext W2oo W2fut.')
    # parser.add_argument('--env', type=str, required=False,  default='CartPole-v0', help='Gym environment. Check available environments in envs.load_environments.env_dict [CartPole-v0].')
    # parser.add_argument('--nn_act', type=str, default='relu', help='Activation function for feed-forward netwroks (relu/tanh) [relu]')
    # parser.add_argument('--method', type=str, required=False, help='function to call.')
    # parser.add_argument('--tfile', type=str, default='results', help='Directory containing test data.')
    # parser.add_argument('--exp_trajs', type=str, help='pickle file with exploration trajectories')
    # parser.add_argument('--p_obs_fail', type=float, help='sensor failure probability for observation.')
    # parser.add_argument('--T_obs_fail', type=int, help='max sensor failure time window for observation.')
    # parser.add_argument('--nh', nargs='+', type=int, default=[16],  help='number of hidden units. --nh L1 L2 ... number of hidden units for each layer [16].')
    # parser.add_argument('--nL', type=int, default=1, help='number of layers. 0- for linear [1]')
    # parser.add_argument('--dbg_len', nargs=2, type=int, help='Specifies a range of traj lengths to be chosen at random')
    # add_boolean_option(parser, 'fullobs', default=False, help='Use fully observable environment state [False].')
    # add_boolean_option(parser, 'normalize_act', default=True, help='Scale actions within bounds [True].')
    # add_boolean_option(parser, 'random_start', default=False, help='Start the psr with Random parameters [False]')
    # add_boolean_option(parser, 'dbg_nobatchpsr', false_name='dbg_batchpsr', default=True, help='Do not use batched PSR updates')
    return parser







