'''
Translates a source file using a translation model.
'''
import argparse

import numpy
import cPickle as pkl

from cmu import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams)

from multiprocessing import Process, Queue

import os
import os.path as osp


def translate_model(queue, rqueue, pid, model, options, k, normalize, n_best):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    from theano import shared
    trng = RandomStreams(1234)
    use_noise = shared(numpy.float32(0.))

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)

    def _translate(seq, pi):
        # sample given an input sequence and obtain scores
        #print(pi.shape, len(seq))
        sample, score = gen_sample(tparams, f_init, f_next,
                numpy.array(seq).reshape([len(seq), 1]),pi[:,None,:],
                                   options, trng=trng, k=k, maxlen=200,
                                   stochastic=False, argmax=False)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        if n_best > 1:
            sidx = numpy.argsort(score)[:n_best]
        else:
            sidx = numpy.argmin(score)
        return numpy.array(sample)[sidx], numpy.array(score)[sidx]

    while True:
        print pid, "get start"
        req = queue.get()
        print pid, "yay"
        if req is None:
            print pid, "break!!!"
            break

        idx, x, pi = req[0], req[1], req[2]
        print pid, '-', idx
        seq, scores = _translate(x, pi)

        rqueue.put((idx, seq, scores))

    return


def main(model, dictionary, dictionary_target, source_file, image_file, rcnn_feats, rcnn_class, image_basedir, saveto, k=5,
         normalize=False, n_process=5, chr_level=False, n_best=1):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # create input and output queues for processes
    queue = Queue()
    rqueue = Queue()
    processes = [None] * n_process
    for midx in xrange(n_process):
        processes[midx] = Process(
            target=translate_model,
            args=(queue, rqueue, midx, model, options, k, normalize, n_best))
        processes[midx].start()

    def get_index(image_basedir,cls,top_n):
        with open(os.path.join(image_basedir,cls.strip()),'r') as f:
            result = []
            idx_result = []
            line = f.readlines()
            for idx,c in enumerate(line):
                if idx==top_n:
                    break
                if c not in result:
                    result.append(c)
                    idx_result.append(idx)
        return idx_result

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fname,image_f,rcnn_feats,rcnn_class,image_basedir):
        with open(fname, 'r') as f, open(rcnn_feats, 'r') as f1, open(rcnn_class, 'r') as f2:
            idx = 0
            images = numpy.load(image_f)
            for line,rcnn_path,cls_path,image in zip(f,f1,f2,images):
                if chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words'] else 1, x)
                x += [0]
                image = image[numpy.newaxis,:]
                idx_ = get_index(image_basedir,cls_path,4)
                rcnn = numpy.load(os.path.join(image_basedir,rcnn_path.strip()))[idx_]
                pi = numpy.concatenate((image,rcnn), axis=0)
                queue.put((idx, x, pi))
                idx += 1
        return idx
                

    def _finish_processes():
        for midx in xrange(n_process):
            queue.put(None)

    def _retrieve_jobs(n_samples):
        trans = [None] * n_samples
        scores = [None] * n_samples
        for idx in xrange(n_samples):
            resp = rqueue.get()
            trans[resp[0]] = resp[1]
            scores[resp[0]] = resp[2]
            if numpy.mod(idx, 10) == 0:
                print 'Sample ', (idx+1), '/', n_samples, ' Done'
        return trans, scores

    print 'Translating ', source_file, '...'
    n_samples = _send_jobs(source_file, image_file,rcnn_feats,rcnn_class,image_basedir )
    trans, scores = _retrieve_jobs(n_samples)
    _finish_processes()

    if n_best == 1:
        trans = _seqs2words(trans)
    else:
        n_best_trans = []
        for idx, (n_best_tr, score_) in enumerate(zip(trans, scores)):
            sentences = _seqs2words(n_best_tr)
            for ids, trans_ in enumerate(sentences):
                n_best_trans.append(
                    '|||'.join(
                        ['{}'.format(idx), trans_,
                         '{}'.format(score_[ids])]))
        trans = n_best_trans

    with open(saveto, 'w') as f:
        print >>f, '\n'.join(trans)
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5, help="Beam size")
    parser.add_argument('-p', type=int, default=5, help="Number of processes")
    parser.add_argument('-n', action="store_true", default=False,
                        help="Normalize wrt sequence length")
    parser.add_argument('-c', action="store_true", default=False,
                        help="Character level")
    parser.add_argument('-b', type=int, default=1, help="Output n-best list")
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('image')
    parser.add_argument('rcnn_feats', type=str)
    parser.add_argument('rcnn_class', type=str)
    parser.add_argument('image_dir', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.model, args.dictionary, args.dictionary_target, args.source,args.image,args.rcnn_feats,args.rcnn_class,args.image_dir,
         args.saveto, k=args.k, normalize=args.n, n_process=args.p,
         chr_level=args.c, n_best=args.b)
