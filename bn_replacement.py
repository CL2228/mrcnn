"""
python bn_replacement.py -e experiment -w 0160
"""
import os
from h5py import File
from optparse import OptionParser

def replace(model_filename, pretrain_filename='checkpoints/pretrained_frcnn.h5'):
    f0, f1 = File(pretrain_filename), File(model_filename)
    layer_names = f0.attrs['layer_names']
    bn_keys = []
    for n in layer_names:
        if n[:2] == b'bn':
            bn_keys.append(n)
    res = f1['resnet_model']
    for k in bn_keys:
        del res[k]
        f0[k].copy(k, res)
    f0.close()
    f1.close()

def copy_and_move(experiment, filename, dest='./reverted_weights',
                  logs='./logs_frcnn', name='coco'):
    os.system('cp {}/matching_mrcnn_{}_{}/matching_mrcnn_{}.h5 {}'.format(
        logs, name, experiment, filename, dest))
    os.system('mv {}/matching_mrcnn_{}.h5 {}/{}_{}.h5'.format(dest, filename, dest, experiment, filename))

usage = 'Author: Zhibo Fan\n Used to revert moving average and moving variance of batch norm layers' \
        'in pretrained resnet as they are not even trained in the following steps.\n However, closing their' \
        'gradients doesn\'t stop the renewage of these two properties.\n' \
        'python bn_replacement.py --weight=\\path\\to\\weight --experiment=exp'
parser = OptionParser(usage)
parser.add_option('-w', '--weight', dest='weight', help='path to weight to be reverted')
parser.add_option('-e', '--experiment', dest='exp', help='experiment name')
parser.add_option('-l', '--logs', dest='logs', help='log file folder name')
parser.add_option('-d', '--dataset', dest='dataset', help='dataset name')
parser.add_option('-f', '--folder', dest='folder', help='where to store the reverted weights')
parser.add_option('-p', '--pretrain', dest='pretrain', help='path to pretrained weights')
options, args = parser.parse_args()

if __name__ == '__main__':
    assert options.exp and options.weight, 'Weight path and experiment must be stated!'
    dest = './convert_weight' if options.folder is None else options.folder
    logs = './logs_frcnn' if options.logs is None else options.logs
    name = 'coco' if options.dataset is None else options.dataset
    pretrain = 'checkpoints/pretrained_frcnn.h5' if options.pretrain is None else options.pretrain
    copy_and_move(options.exp, options.weight, dest=dest, logs=logs, name=name)
    new_weight_path = '{}/{}_{}.h5'.format(dest, options.exp, options.weight)
    replace(new_weight_path, pretrain_filename=pretrain)
    print('\n', new_weight_path, '\n')



