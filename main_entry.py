"""
Tips for

Uses multiple GPUs, indicated by the flag --nr-gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

# utility modules
import os
from os import path
import shutil
import sys
import time
import json
import argparse
import random
import numpy as np
import tensorflow as tf
import tensorlayer as tl

from tqdm import tqdm

ITEM_DIM = 100

# Add the SRW to the environment variable PYTHONPATH
dir_path = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))
sys.path.append(dir_path)

from utils import *

import settings

from pprint import pprint as pr


# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    # meta info
    parser.add_argument('--mode', type=str, default='sample',
                        help='train or sample')

    # data I/O
    parser.add_argument('--model_directory', type=str, default=settings.MODEL_STORE_PATH,
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('--model_name', type=str, default='lstm_model',
                        help='gru_model|lstm_model model file name (will create a separated folder)')
    parser.add_argument('-d', '--data_set', type=str, default='kanji',
                        help='Can be kanji')
    parser.add_argument('-c','--checkpoint_interval', type=int, default=1,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r','--report_interval', type=int, default=1000,
                        help='Every how many epochs to report current situation?')
    parser.add_argument('-v','--validation_interval', type=int, default=10,
                        help='Every how many epochs to do validation current situation?')
    parser.add_argument('--load_params', type=bool, default=False,
                        help='Restore training from previous model checkpoint')
    parser.add_argument('--silent', type=bool, default=False,
                        help='remove progress')
    parser.add_argument('--random_sample', type=bool, default=False,
                        help='train with full data or with random samples')

    # data set construction
    parser.add_argument('--training_num', type=int, default=None,
                        help='number of training samples')
    parser.add_argument('--training_epoch', type=int, default=300,
                        help='number of training epoch')
    parser.add_argument('--val_portion', type=float, default=0.005,
                        help='The portion of data to be validation data')

    # model
    parser.add_argument('--rnn_size', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('-q', '--seq_length', type=int, default=30,
                        help='The minimum length of history sequence')

    # hyper-parameter for optimization
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float,
                        default=0.1, help='Base learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size during training per GPU')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='Dropout strength, where 0 = No dropout, higher = more dropout.')
    parser.add_argument('--sample_dropout_rate', type=float, default=0.5,
                        help='Dropout rate for selecting training samples, where 0 = No dropout, higher = more dropout.')
    parser.add_argument('-g', '--nr_gpu', type=int, default=1,
                        help='The number GPUs to distribute the training across')

    # reproducibility:random seed
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed to use')

    # sample options
    parser.add_argument('--filename', type=str, default='output',
                       help='filename of .svg file to output, without .svg')
    parser.add_argument('--sample_length', type=int, default=600,
                       help='number of strokes to sample')
    parser.add_argument('--picture_size', type=float, default=160,
                       help='a centered svg will be generated of this size')
    parser.add_argument('--scale_factor', type=float, default=1,
                       help='factor to scale down by for svg output.  smaller means bigger output')
    parser.add_argument('--num_picture', type=int, default=20,
                       help='number of pictures to generate')
    parser.add_argument('--num_col', type=int, default=5,
                       help='if num_picture > 1, how many pictures per row?')
    parser.add_argument('--color_mode', type=int, default=1,
                       help='set to 0 if you are a black and white sort of person...')
    parser.add_argument('--stroke_width', type=float, default=2.0,
                       help='thickness of pen lines')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='sampling temperature')

    args = parser.parse_args()
    print('INFO CHECK!\ninput args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))


    ################################################
    #        The main program starts
    ################################################

    # fix random seed for reproducibility
    # rng = np.random.RandomState(args.seed)
    print("random seed is",args.seed)
    # tf.set_random_seed(args.seed)
    if args.mode=='train':
        train(args)
    elif args.mode=='sample':
        sample(args)

def train(args):
    class_num = {'ml_100K': 1682,
                 'ml_1M': 3952,
                 'kanji':5,
                 'netflix': 17770}[args.data_set]

    # initialize data loaders for train/test splits
    args.class_num = class_num

    if args.data_set == 'kanji':
        print('start loading dataset', args.data_set)
        # data loader
        import data.kanjivg_data as data
        dataLoader = data.DataLoader(args)
        print('dataset', args.data_set, 'loading completed')
    else:
        print('this dataset is not available , or the dataset name not correct')
        quit()

    args.model_file_name=args.model_name+"_file"
    if args.model_name == "graves_model":
        from leaner_model.graves_model import LSTM_Model_Session as model_session
        print('import graves model LSTM model okay')
    else:
        print('not valid name for model')
        quit()

    model_path_name = path.join(args.model_directory, args.model_file_name)
    print("The model's path is", model_path_name)

    if args.load_params is True and os.path.exists(model_path_name):
        try:
            model = model_session.restore(model_path_name)
        except:
            print("error happens, now remove original path folder", model_path_name)
            shutil.rmtree(model_path_name)
            os.makedirs(model_path_name)
            model = model_session.create(class_num=class_num, item_dim=ITEM_DIM, args=args)
    else:
        if not os.path.isdir(model_path_name):
            os.makedirs(model_path_name)
            print("The folder to store the model has been created")
        print("Not load prestored model, creating a new model ...")
        model = model_session.create(class_num=class_num, item_dim=ITEM_DIM, args=args)
    print(model)


    args.max_len=dataLoader.max_len
    model.args=args
    model.dataLoader=dataLoader

    if args.training_num is None:
        args.training_num = len(dataLoader.X_train)

    X = dataLoader.X_train[:args.training_num]
    Y = dataLoader.y_train[:args.training_num]

    for iEpoch in range(args.training_epoch):
        print("epoch num",iEpoch)

        for x, y in tqdm(tl.iterate.minibatches(X, Y, args.batch_size, shuffle=True), disable=args.silent):
            # Train the model
            iteration = model.train(x, y,  args.learning_rate, args.dropout_rate)
        if (iEpoch+1) % args.report_interval == 0:
            training_batch_accuracy = model.test_batch(X[:args.batch_size], Y[:args.batch_size])
            print("%s: training batch accuracy %0.4f" % (model, training_batch_accuracy))
        if (iEpoch+1) % args.validation_interval == 0:
            validation_acc = model.test(dataLoader.X_val, dataLoader.y_val)
            print("%s: validataion acc: %0.4f" % (model, validation_acc))
        if (iEpoch+1) % args.checkpoint_interval == 0:
            model.save(model_path_name)

    print("Final model %s" % model)


def sample(args):
    model_path_name = path.join(args.model_directory, args.model_name)

    print("start loading data set",args.data_set)

    import data.kanjivg_data as data
    data_loader = data.SketchLoader(args)

    print("data set loading okay")
    input_data, target_data = data_loader.next_batch()
    print(input_data.shape)
    print(target_data.shape)

    quit()
    draw_sketch_array(input_data, args, svg_only = True)

    quit()

    # TODO:get data

    model = Model_Session.restore(model_path_name)
    print(model)
    accuracy = model.test(test_data.images, test_data.labels)
    accuracy_top_5 = model.test_top_n(test_data.images, test_data.labels, n=5)
    accuracy_top_10 = model.test_top_n(test_data.images, test_data.labels, n=10)
    accuracy_top_15 = model.test_top_n(test_data.images, test_data.labels, n=15)
    accuracy_top_20 = model.test_top_n(test_data.images, test_data.labels, n=20)

    print("Test accuracy %0.4f" % accuracy)
    print("top-5 : %f" % accuracy_top_5)
    print("top-10 : %f" % accuracy_top_10)
    print("top-15 : %f" % accuracy_top_15)
    print("top-20 : %f" % accuracy_top_20)

if __name__ == "__main__":
    main()
