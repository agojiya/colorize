""" Adapted from mnist-digit-classification """
import os


def get_highest_save_file(save_path):
    """ Returns the epoch and data-set index of the latest save file. """
    files = [file for file in os.listdir(save_path) if
             isvalid(save_path, file)]
    highest_epoch, highest_index = 0, 0
    for file in files:
        epoch = int(str(file).split('-')[-2])
        index = int(str(file).split('-')[-1].replace('.meta', ''))
        if epoch > highest_epoch:
            highest_epoch = epoch
            highest_index = 0
        if epoch == highest_epoch and index > highest_index:
            highest_index = index
    return highest_epoch, highest_index


def isvalid(save_path, file):
    """ Returns true if the file described by the parameters is a file with
    the appropriate file extension. """
    return os.path.isfile(os.path.join(save_path, file)) and \
        str(file).endswith('.meta')
