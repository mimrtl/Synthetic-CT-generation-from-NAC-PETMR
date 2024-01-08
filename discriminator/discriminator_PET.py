#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import datetime
import argparse
from global_dict.w_global import gbl_set_value
from model_PET.w_train import train_a_vgg


def usage():
    print("Error in input argv")


def main():
    parser = argparse.ArgumentParser(
        description='''This is for discriminating PET images and CT images for perceptual loss calculation (1st round). ''',
        epilog="""All's well that ends well.""")
    parser.add_argument('--slice_x', metavar='', type=int, default=1,
                        help='Slices of input(1)<int>[1/3]')
    parser.add_argument('--color_mode', metavar='', type=int, default=1,
                        help='Greyscale(1) or RGB(3)<int>[1/3]')

    parser.add_argument('--size_x', metavar='', type=int, default=64,
                        help='chunk size: x dimension')
    parser.add_argument('--size_y', metavar='', type=int, default=64,
                        help='chunk size: y dimension')
    parser.add_argument('--size_z', metavar='', type=int, default=32,
                        help='chunk size: z dimension')

    parser.add_argument('--stride_x', metavar='', type=int, default=8,
                        help='stride of the x dimension')
    parser.add_argument('--stride_y', metavar='', type=int, default=8,
                        help='stride of the y dimension')
    parser.add_argument('--stride_z', metavar='', type=int, default=8,
                        help='stride of the z dimension')

    parser.add_argument('--id', metavar='', type=str, default="discriminator",
                        help='ID of the current model.(eeVee)<str>')
    parser.add_argument('--epoch', metavar='', type=int, default=500,
                        help='Number of epoches of training(300)<int>')
    parser.add_argument('--batch_size', metavar='', type=int, default=96,
                        help='batch size of training(10)<int>')
    parser.add_argument('--gp_id', metavar='', type=str, default='5',
                        help='group id used for model train <str>')
    parser.add_argument('--train_path', metavar='', type=str, default='../groups/gp_5/train/norm_nii/',
                        help='The training dataset path <str>')
    parser.add_argument('--val_path', metavar='', type=str, default='../groups/gp_5/val/norm_nii/',
                        help='The validation dataset path <str>')

    parser.add_argument('--pretrained_path', metavar='', type=str, default='',
                        help='The pretrained model path <str>')

    parser.add_argument('--method', metavar='', type=str, default='per_vgg',
                        help='loss used for model training <str>')
    parser.add_argument('--rounds', metavar='', type=str, default='1st_round',
                        help='1st round or 2nd round <str>')

    args = parser.parse_args()

    dir_syn = '../result/gp_' + args.gp_id + '/' + str(args.size_x) + '-' + str(args.size_y) + '-' + str(args.size_z) + '/unet/' + str(args.method) + '/' + str(args.rounds) + '/'
    dir_model = '../result/gp_' + args.gp_id + '/' + str(args.size_x) + '-' + str(args.size_y) + '-' + str(args.size_z) + '/unet/' + str(args.method) + '/' + str(args.rounds) + '/'

    if not os.path.exists(dir_syn):
        os.makedirs(dir_syn)

    if not os.path.exists(dir_model):
        os.makedirs(dir_model)

    time_stamp = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")
    model_id = args.id + time_stamp
    gbl_set_value("gp_id", args.gp_id)
    gbl_set_value("dir_syn", dir_syn)
    gbl_set_value("dir_model", dir_model)
    gbl_set_value("model_id", model_id)
    gbl_set_value("n_epoch", args.epoch + 1)
    gbl_set_value("batch_size", args.batch_size)
    gbl_set_value("color_mode", args.color_mode)
    gbl_set_value("slice_x", args.slice_x)
    gbl_set_value("size_x", args.size_x)
    gbl_set_value("size_y", args.size_y)
    gbl_set_value("size_z", args.size_z)
    gbl_set_value("stride_x", args.stride_x)
    gbl_set_value("stride_y", args.stride_y)
    gbl_set_value("stride_z", args.stride_z)
    gbl_set_value("pretrained_path", args.pretrained_path)
    gbl_set_value("train_path", args.train_path)
    gbl_set_value("val_path", args.val_path)
    gbl_set_value("method", args.method)
    gbl_set_value("rounds", args.rounds)

    # check training and validation path
    print(args.train_path)
    print(args.val_path)

    # start training
    model = train_a_vgg(pretrained_model=0)
    print("Training Completed!")


if __name__ == "__main__":
    main()
