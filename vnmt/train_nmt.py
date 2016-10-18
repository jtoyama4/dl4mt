import numpy
import os

from nmt import train


def main(job_id, params):
    print params
    basedir = '/home/ubuntu/dl4mt-tutorial'
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
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
                     dictionaries=['%s/flickr30k/bitext.train.en.tok.txt.pkl' % basedir,
                                   '%s/flickr30k/bitext.train.de.tok.txt.pkl' % basedir],
                     validFreq=1000,
                     dispFreq=1,
                     saveFreq=1000,
                     sampleFreq=50,
                     use_dropout=params['use-dropout'][0],
                     overwrite=False)
    return validerr

if __name__ == '__main__':
    basedir = '/home/ubuntu/dl4mt-tutorial'
    main(0, {
        'model': ['%s/models/model_flickr30k.npz' % basedir],
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
        'reload': [False]})
