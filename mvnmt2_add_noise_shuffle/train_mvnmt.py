import numpy
import os
import os.path as osp
import argparse

print "This is MVNMT"

from mvnmt import train


def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     fine_tuning=params['fine_tuning'][0],
                     fine_tuning_load=params['fine_tuning_load'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     dimv=params['dimv'][0],
                     dim_pi=params['dim_pi'][0],
                     dim_pic=params['dim_pic'][0],
                     n_words=params['n-words'][1],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     maxlen=params['maxlen'],
                     batch_size=params['batchsize'],
                     valid_batch_size=params['batchsize'],
                     datasets=['%s/flickr30k/bitext.train.en.tok.txt' % params['basedir'],
                               '%s/flickr30k/bitext.train.de.tok.bpe.txt' % params['basedir'],
                               '%s/flickr30k/fc7.train.npy' % params['basedir']],
                     valid_datasets=['%s/flickr30k/bitext.val.en.tok.txt' % params['basedir'],
                                     '%s/flickr30k/bitext.val.de.tok.bpe.txt' % params['basedir'],
                                     '%s/flickr30k/fc7.val.npy' % params['basedir']],
                     dictionaries=['%s/flickr30k/bitext.train.en.tok.txt.pkl' % params['basedir'],
                                   '%s/flickr30k/bitext.train.de.tok.bpe.txt.pkl' % params['basedir']],
                     validFreq=1000,
                     dispFreq=100,
                     saveFreq=1000,
                     sampleFreq=500,
                     use_dropout=params['use-dropout'][0],
                     overwrite=False)
    return validerr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--load',action='store_true',default=False)
    parser.add_argument('--fine_tuning', action='store_true', default=True)
    parser.add_argument('--modeldir', type=str, default="mvnmt2")
    parser.add_argument('--finetunedir', type=str, default="nmt")
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--dim_word', type=int, default=256)
    parser.add_argument('--dimv', type=int, default=256)
    parser.add_argument('--dim_pi', type=int, default=4096)
    parser.add_argument('--dim_pic', type=int, default=256)
    parser.add_argument('--dim_enc_z', type=int, default=256)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--decay_c', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=1.0)
    args = parser.parse_args()

    basedir = osp.join(osp.dirname(osp.abspath(__file__)), "../")
    modeldir = osp.join(basedir, "models", args.modeldir)
    finetunedir = osp.join(basedir, "models", args.finetunedir)
    validdir = osp.join(modeldir, "valid")
    scriptdir = osp.join(basedir, "script")
    print("basedir: {}".format(basedir))
    print("modeldir: {}".format(modeldir))
    if not osp.exists(modeldir):
        os.makedirs(modeldir)
    if not osp.exists(validdir):
        os.makedirs(validdir)

    main(0, {
        'basedir': basedir,
        'model': ['%s/model_mvnmt2.npz' % modeldir],
        'fine_tuning_load':['%s/model_nmt.npz' % finetunedir],
        'validdir': validdir,
        'scriptdir': scriptdir,
        'dim_word': [args.dim_word],
        'dim': [args.dim],
        'dimv': [args.dimv],
        'dim_pi': [args.dim_pi],
        'dim_pic': [args.dim_pic],
        'dim_enc_z': [args.dim_enc_z],
        #'n-words': [30000],
        'n-words': [10211,13180],
        'optimizer': ['adadelta'],
        'decay-c': [args.decay_c],
        'clip-c': [1.],
        'use-dropout': [False],
        #'learning-rate': [0.0001],
        'learning-rate': [args.lr],
        'reload': [args.load],
        'batchsize': args.batchsize,
        'maxlen': args.maxlen,
        'fine_tuning': [args.fine_tuning]})

