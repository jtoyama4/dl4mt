import numpy
import os
import argparse

print "This is MVNMT"

from vrnn_captioning import train


def main(job_id, params):
    print params
    basedir = '/home/ubuntu/dl4mt-tutorial'
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     fine_tuning=params['fine_tuning'][0],
                     fine_tuning_load=params['fine_tuning_load'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     dimv=params['dimv'][0],
                     dim_pi=params['dim_pi'][0],
                     dim_pic=params['dim_pic'][0],
                     dim_l=params['dim_l'][0],
                     n_words=params['n-words'][1],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     maxlen=30,
                     batch_size=128,
                     valid_batch_size=64,
                     datasets=['%s/flickr30k/s_caption_train.txt' % basedir,
                               '%s/flickr30k/kelvin_feature/train-cnn.npy' % basedir],
                     valid_datasets=['%s/flickr30k/bitext.val.en.tok.txt' % basedir,
                                     '%s/flickr30k/kelvin_feature/dev-cnn.npy' % basedir],
                     dictionaries=['%s/flickr30k/caption_train.txt.pkl' % basedir],
                     validFreq=100,
                     dispFreq=1,
                     saveFreq=1000,
                     sampleFreq=50,
                     use_dropout=params['use-dropout'][0],
                     overwrite=False)
    return validerr

if __name__ == '__main__':
    basedir = '/home/ubuntu/dl4mt-tutorial'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--load',action='store_true',default=False)
    parser.add_argument('--fine_tuning', action='store_true', default=False)
    args = parser.parse_args()

    main(0, {
        'model': ['%s/models/vrnn/model_vrnn_captioning.npz' % basedir],
        'fine_tuning_load':['%s/models/vrnn/model_vnmt.npz' % basedir],
        'dim_word': [1000],
        'dim': [1000],
        'dimv': [250],
        'dim_l': [500],
        'dim_pi': [512],
        'dim_pic': [200],
        #'n-words': [30000],
        'n-words': [10211,18723],
        'optimizer': ['adam'],
        'decay-c': [0.0001],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.001],
        #'learning-rate': [1.0],
        'reload': [args.load],
        'fine_tuning': [args.fine_tuning]})
