import numpy
import os
import os.path as osp
import argparse

print "This is VNMT"

basedir = osp.join(osp.dirname(osp.abspath(__file__)), "../")
modeldir = osp.join(basedir, "models", "vnmt_no_finetune")
finetunedir = osp.join(basedir, "models", "nmt")
validdir = osp.join(modeldir, "valid")
scriptdir = osp.join(basedir, "script")
print("basedir: {}".format(basedir))
print("modeldir: {}".format(modeldir))
if not osp.exists(modeldir):
    os.makedirs(modeldir)
if not osp.exists(validdir):
    os.makedirs(validdir)

from vnmt import train


def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     fine_tuning=params['fine_tuning'][0],
                     fine_tuning_load=params['fine_tuning_load'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     dimv=params['dimv'][0],
                     n_words=params['n-words'][1],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     maxlen=30,
                     batch_size=128,
                     valid_batch_size=128,
                     datasets=['%s/flickr30k/bitext.train.en.tok.txt' % basedir,
                               '%s/flickr30k/bitext.train.de.tok.txt' % basedir],
                     valid_datasets=['%s/flickr30k/bitext.val.en.tok.txt' % basedir,
                                     '%s/flickr30k/bitext.val.de.tok.txt' % basedir],
                     valid_detok_datasets=['%s/flickr30k/val.norm.ln.en' % basedir,
                                     '%s/flickr30k/val.norm.ln.de' % basedir],
                     dictionaries=['%s/flickr30k/bitext.train.en.tok.txt.pkl' % basedir,
                                   '%s/flickr30k/bitext.train.de.tok.txt.pkl' % basedir],
                     validFreq=1000,
                     dispFreq=1,
                     saveFreq=1000,
                     sampleFreq=50,
                     use_dropout=params['use-dropout'][0],
                     validdir=params['validdir'],
                     scriptdir=params['scriptdir'],
                     overwrite=False)
    return validerr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--load',action='store_true',default=False)
    parser.add_argument('--fine_tuning', action='store_true', default=False)
    args = parser.parse_args()

    main(0, {
        'model': ['%s/model_vnmt_no_finetune.npz' % modeldir],
        'fine_tuning_load':['%s/model_nmt.npz' % finetunedir],
        'validdir': validdir,
        'scriptdir': scriptdir,
        'dim_word': [256],
        'dim': [256],
        'dimv': [100],
        #'n-words': [30000],
        'n-words': [10211,18723],
        'optimizer': ['adadelta'],
        'decay-c': [0.0001],
        'clip-c': [1.],
        'use-dropout': [False],
        #'learning-rate': [0.0001],
        'learning-rate': [1.0],
        'reload': [args.load],
        'fine_tuning': [args.fine_tuning]})
