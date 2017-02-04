'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

from data_iterator import TextIterator

profile = False


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj

# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'variation': ('param_init_variation', 'variation_layer'),
          'variational':('param_init_variational','variational_layer'),
          'variational_gru':('param_init_variational_gru','variational_gru_layer'),
          'image_attention':('param_init_image_attention','image_attention_layer')
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# batch preparation
def prepare_data(seqs_x, images=None, maxlen=None, n_words_src=30000, n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    #lengths_y = [len(s) for s in seqs_y]
    # pi: a list of images (batch_size,seq_length,dim)
    if maxlen is not None:
        new_seqs_x = []
        #new_seqs_y = []
        new_lengths_x = []
        #new_lengths_y = []
        new_images = []

        for l_x, s_x, p in zip(lengths_x, seqs_x, images):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                #new_seqs_y.append(s_y)
                #new_lengths_y.append(l_y)
                new_images.append(p)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        #lengths_y = new_lengths_y
        #seqs_y = new_seqs_y
        images = new_images

        if len(lengths_x) < 1:
            return None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    #maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    #y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    #y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        #y[:lengths_y[idx], idx] = s_y
        #y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, images


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


def param_init_image_attention(options, params, prefix="image_attention", nin_pi=None, nin_h=None, nout=None, ortho=True):
    if nin_pi is None:
        nin_pi = options['dim_pi']
    if nin_h is None:
        nin_h = options['dim']
    if nout is None:
        nout = options["dim_pic"]

    params[_p(prefix, 'W')] = norm_weight(nin_h, nout, scale=0.01)
    params[_p(prefix, 'U')] = norm_weight(nin_pi, nout, scale=0.01)
    params[_p(prefix, 'b')] = norm_weight(nout, 1, scale=0.01)

    return params

def image_attention_layer(tparams, state_below, options, prefix='image_attention', **kwargs):
    #state_below[0] is image and state_below[1] is previous hidden state in encoder.
    U = tparams[0]
    W = tparams[1]
    b = tparams[2]
    p_i = tensor.dot(state_below[0], U)
    p_h = tensor.dot(state_below[1], W)
    if p_i.ndim == 3:
        p_ih = p_i + p_h[:,None,:]
    else:
        p_i = p_i.reshape((1,p_i.shape[0], p_i.shape[1]))
        p_ih = p_i + p_h[:,None,:]
    e = tensor.dot(tensor.tanh(p_ih), b)
    alpha_shp = e.shape
    alpha = tensor.nnet.softmax(e.reshape([alpha_shp[0], alpha_shp[1]]))
    """if p_i.ndim == 3:
        ctx = tensor.sum(state_below[0] * alpha[:,:,None], axis=1)
    else:
        alpha_sample = alpha.reshape((alpha.shape[0],))
        ctx = tensor.sum(state_below[0] * alpha_sample[:,None], axis=0) 
    """
    ctx = tensor.sum(state_below[0] * alpha[:,:,None], axis=1)
    return ctx
    
#variation layer
def param_init_variational(options, params, prefix='variational', nin=None, dim=None, dimv=None, dim_l=None):
    if nin is None:
        nin = optios['dim_word']
    if dim is None:
        dim = options['dim']
    if dimv is None:
        dimv = options['dimv']
    if dim_l is None:
        dim_l = options['dim_l']
    
    #prior network using mlp
    W_pri_1 = numpy.concatenate([norm_weight(options['dim_pi'],dim_l),norm_weight(dim,dim_l)])
    params[_p(prefix, 'W_pri_1')] = W_pri_1
    params[_p(prefix, 'W_pri_1_b')] = numpy.zeros((dim_l,)).astype('float32')

    W_pri_2 = norm_weight(dim_l, dim_l)
    params[_p(prefix, 'W_pri_2')] = W_pri_2
    params[_p(prefix, 'W_pri_2_b')] = numpy.zeros((dim_l,)).astype('float32')
    
    W_pri_mu = norm_weight(dim_l,dimv)
    params[_p(prefix, 'W_pri_mu')] = W_pri_mu
    params[_p(prefix, 'W_pri_mu_b')] = numpy.zeros((dimv,)).astype('float32')

    W_pri_sigma = norm_weight(dim_l, dimv)
    params[_p(prefix, 'W_pri_sigma')] = W_pri_sigma
    params[_p(prefix, 'W_pri_sigma_b')] = numpy.zeros((dimv,)).astype('float32')
    
    #post network using mlp
    W_post_1 = numpy.concatenate([norm_weight(options['dim_pi'],dim_l),norm_weight(dim,dim_l),norm_weight(nin,dim_l)])
    params[_p(prefix, 'W_post_1')] = W_post_1
    params[_p(prefix, 'W_post_1_b')] = numpy.zeros((dim_l,)).astype('float32')

    W_post_2 = norm_weight(dim_l, dim_l)
    params[_p(prefix, 'W_post_2')] = W_post_2
    params[_p(prefix, 'W_post_2_b')] = numpy.zeros((dim_l,)).astype('float32')
    
    W_post_mu = norm_weight(dim_l,dimv)
    params[_p(prefix, 'W_post_mu')] = W_post_mu
    params[_p(prefix, 'W_post_mu_b')] = numpy.zeros((dimv,)).astype('float32')

    W_post_sigma = norm_weight(dim_l,dimv)
    params[_p(prefix, 'W_post_sigma')] = W_post_sigma
    params[_p(prefix, 'W_post_sigma_b')] = numpy.zeros((dimv,)).astype('float32')
    
    #encoding z (phi tau z)
    W_z_1 = norm_weight(dimv,dim_l)
    params[_p(prefix, "W_z_1")] = W_z_1

    W_z_2 = norm_weight(dim_l,dim_l)
    params[_p(prefix, "W_z_2")] = W_z_2

    #generate x (phi tau dec)
    W_dec_1 = numpy.concatenate((norm_weight(dim_l,dim_l), norm_weight(dim,dim_l), norm_weight(options['dim_word'],dim_l)),axis=0)
    params[_p(prefix, 'W_dec_1')] = W_dec_1
    params[_p(prefix, 'W_dec_1_b')] = numpy.zeros((dim_l,)).astype('float32')

    W_dec_2 = norm_weight(dim_l, dim_l)
    params[_p(prefix, 'W_dec_2')] = W_dec_2
    params[_p(prefix, 'W_dec_2_b')] = numpy.zeros((dim_l,)).astype('float32')

    #W_dec_sigma = numpy.concatenate((norm_weight(dimv,nin), norm_weight(dim,nin)),axis=0)
    #params[_p(prefix, 'W_dec_sigma')] = W_dec_sigma
    #params[_p(prefix, 'W_dec_sigma_b')] = numpy.zeros((nin,)).astype('float32')
    
    #decoder can be categorical (if try to directly generate word)
    #W_dec_mu = norm_weight(nin,nin)
    #params[_p(prefix, 'W_dec_mu')] = W_dec_mu                    
    #params[_p(prefix, 'W_dec_mu_b')] = numpy.zeros((nin,)).astype('float32')
    
    W_dec_mu = norm_weight(dim_l, options['n_words_src'])
    params[_p(prefix, 'W_dec_mu')] = W_dec_mu
    params[_p(prefix, 'W_dec_mu_b')] = numpy.zeros((options['n_words_src'],)).astype('float32')

    return params

def variational_layer(tparams, state_below, options, prefix='variational', mask=None, training=True, nsteps = None,**kwargs):
    ctxpi = state_below[0]
    h_ = state_below[1]
    #nsteps = h_.shape[0]
    """if ctxpi.ndim == 2:
        training = True
    """
        
    if training:
        _x = state_below[2]
        prev_x = state_below[3]
    else:
        prev_x = state_below[2]

    W_pri_1 = tparams[0]
    W_pri_1_b = tparams[1]
    W_pri_2 = tparams[2]
    W_pri_2_b = tparams[3]
    W_pri_mu = tparams[4]
    W_pri_mu_b = tparams[5]
    W_pri_sigma = tparams[6]
    W_pri_sigma_b = tparams[7]

    W_post_1 = tparams[8]
    W_post_1_b = tparams[9]
    W_post_2 = tparams[10]
    W_post_2_b = tparams[11]
    W_post_mu = tparams[12]
    W_post_mu_b = tparams[13]
    W_post_sigma = tparams[14]
    W_post_sigma_b = tparams[15]

    W_z_1 = tparams[16]
    W_z_2 = tparams[17]
    W_dec_1 = tparams[18]
    W_dec_1_b = tparams[19]
    W_dec_2 = tparams[20]
    W_dec_2_b = tparams[21]
    W_dec_mu = tparams[22]
    W_dec_mu_b = tparams[23]
    
    #prior
    if training:
        pri_1 = tensor.dot(concatenate([ctxpi,h_],axis=1),W_pri_1) + W_pri_1_b
        pri_2 = tensor.dot(pri_1, W_pri_2) + W_pri_2_b
        pri_mu = tensor.dot(pri_2, W_pri_mu) + W_pri_mu_b
        pri_sigma = tensor.nnet.softplus(tensor.dot(pri_2,W_pri_sigma) + W_pri_sigma_b)

        post_1 = tensor.dot(concatenate([ctxpi,h_,_x], axis=1), W_post_1) + W_post_1_b
        post_2 = tensor.dot(post_1, W_post_2) + W_post_2_b
        post_mu = tensor.dot(post_2, W_post_mu) + W_post_mu_b
        post_sigma = tensor.nnet.softplus(tensor.dot(post_2, W_post_sigma) + W_post_sigma_b)
    else:
        #assert h_.ndim 
        #ctxpi = ctxpi.reshape((1,ctxpi.shape[0]))
        pri_1 = tensor.dot(concatenate([ctxpi,h_],axis=1),W_pri_1) + W_pri_1_b
        pri_2 = tensor.dot(pri_1, W_pri_2) + W_pri_2_b
        pri_mu = tensor.dot(pri_2, W_pri_mu) + W_pri_mu_b
        pri_sigma = tensor.nnet.softplus(tensor.dot(pri_2,W_pri_sigma) + W_pri_sigma_b)

    #KL computing
    kl_cost = tensor.alloc(0.,1)
    epsilon = 10**(-8)
    if training:
        kl = theano.tensor.log(pri_sigma / post_sigma) + ((post_sigma**2) + ((post_mu - pri_mu)**2)) / (epsilon + 2 * pri_sigma**2) - 0.5
        kl_cost = tensor.sum(kl)

    #generate x
    def _gaussian_noise_step(mu, sigma, noise, z, add_noise=True):
        if not add_noise:
            return mu
        else:
            SIGMA = tensor.diag(sigma)
            result = mu + tensor.dot(SIGMA, noise)
            return result

    trng = RandomStreams(1234)
    dimv = options['dimv']
    
    normal_noise = trng.normal((dimv,))
    if training:
        normal_noise = trng.normal((nsteps, dimv))

    seqs = [pri_mu, pri_sigma, normal_noise]
    if not training:
        seqs = [pri_mu, pri_sigma, normal_noise]
    
    sample_func = lambda m,s,n,z: _gaussian_noise_step(m,s,n,z,add_noise=False)
    if training:
        sample_func = lambda m,s,n,z: _gaussian_noise_step(m,s,n,z,add_noise=True)

    sample_z, _ = theano.scan(sample_func,
                    sequences=seqs,
                              outputs_info=[tensor.alloc(0.,dimv)],
                              name="variation_z_%s" % prefix,
                              n_steps = nsteps)
    assert sample_z != None, 'man , sample z is NONE!!'    
    
    encoded_z = tensor.dot(sample_z, W_z_1)
    encoded_z = tanh(tensor.dot(encoded_z, W_z_2))
    #generate_mu = tensor.dot(concatenate([encoded_z, h_],axis=1) , W_dec_mu) + W_dec_mu_b
    
    
    if training:
        dec_1 = tensor.dot(concatenate([encoded_z, h_, prev_x],axis=1), W_dec_1) + W_dec_1_b
        dec_2 = tensor.dot(dec_1, W_dec_2) + W_dec_2_b
    else:
        dec_1 = tensor.dot(concatenate([encoded_z, h_, prev_x],axis=1), W_dec_1) + W_dec_1_b
        dec_2 = tensor.dot(dec_1, W_dec_2) + W_dec_2_b
    
    x_mu = tensor.dot(dec_2, W_dec_mu) + W_dec_mu_b
    x_prob = tensor.nnet.softmax(x_mu)
    """if training:
        next_word = None
    else:
        next_word = x_prob.argmax()"""
    return kl_cost, sample_z, x_prob

#GRU layer
def param_init_variational_gru(options, params, prefix='variational_gru', nin=None, dim=None, dimv=None):
    if nin is None:
        nin = optios['dim_proj']
    if dim is None:
        dim = options['dim_proj']
    if dimv is None:
        dimv = options['dimv']


    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # gru parameters for z
    Z = numpy.concatenate([norm_weight(dimv, dim), norm_weight(dimv, dim)], axis=1)
    params[_p(prefix, 'Z')] = Z


    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    Zx = norm_weight(dimv, dim)
    params[_p(prefix, 'Zx')] = Zx

    #parameters to initialize h0
    #W_init = norm_weight(options['dim_pi'],options['dim'])
    #params[_p(prefix, 'W_init')] = W_init
    return params

def variational_gru_layer(tparams, state_below, options, prefix='variational_gru', mask=None, one_step=False,init_state=None, prev_x=None,**kwargs):
    #pis (batch, 196, 512)
    #xs  (seq, batch, nin)
    if one_step:
        assert init_state, 'init_state must be provided when sampling'
        assert prev_x, 'prev x must be provided when sampling'

    pi = state_below[0]
    x = state_below[1]
    shift_x = tensor.zeros_like(x)
    shift_x = tensor.set_subtensor(shift_x[1:], x[:-1])
    
    if one_step:
        n_samples = 1
    else:
        n_samples = x.shape[1]


    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., x.shape[0], 1)

    #utility function to slice a tensor
    def _slice(_x,n,dim):
        if _x.ndim == 3:
            return _x[:,:,n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(x, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(x, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    if not one_step:
        mean_pi = tensor.mean(pi, axis=1)
        h0 = get_layer('ff')[1](tparams, mean_pi, options, prefix='ff_init_state', activ='tanh')
    
    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x, prevx, x_, xx_, h_, kl, gx, U, Ux, Z, Zx, i0, i1, i2, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, pi, n):
        preact = tensor.dot(h_, U)
        preact += x_
        tparams_i = [i0, i1, i2]
        tparams_v = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23]
        #compute attened image
        ctxpi = get_layer('image_attention')[1](tparams_i,[pi,h_],options=options)
        
        #compute z
        kl, z, g_x = get_layer('variational')[1](tparams_v,[ctxpi, h_, x, prevx],options,mask=m_,nsteps=n, training=True)
        
        preact_z = tensor.dot(z,Z)
        preact += preact_z
        
        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_ + tensor.dot(z,Zx)

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return [h, kl.reshape((1,)), g_x]

    def _step_slice_sample(m_, x_, xx_, h_, gx, U, Ux, Z, Zx, i0, i1, i2, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, pi, n):
        preact = tensor.dot(h_, U)
        preact += x_

        tparams_i = [i0, i1, i2]
        tparams_v = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23]

        #compute attened image
        ctxpi = get_layer('image_attention')[1](tparams_i,[pi,h_],options=options)
        
        #compute z
        kl, z, g_x = get_layer('variational')[1](tparams_v,[ctxpi,h_, gx],options,mask=m_,nsteps=n, training=False)
        
        preact_z = tensor.dot(z,Z)
        preact += preact_z
        
        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_ + tensor.dot(z,Zx)

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return [h, g_x]

    #pi_seq = pis.dimshuffle(,0,1,2)
    seqs = [mask, x, shift_x, state_below_, state_belowx]
    nsteps = x.shape[0]
    nn = x.shape[1]
    if one_step:
        nn = init_state.shape[0]
    if not one_step:
        init_states = [h0, tensor.alloc(0.,1), tensor.alloc(0., n_samples, options['n_words_src'])]
    
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Z')],
                   tparams[_p(prefix, 'Zx')],
                   tparams[_p('image_attention', 'U')],
                   tparams[_p('image_attention', 'W')],
                   tparams[_p('image_attention', 'b')],
                   tparams[_p('variational','W_pri_1')],
                   tparams[_p('variational','W_pri_1_b')],
                   tparams[_p('variational','W_pri_2')],
                   tparams[_p('variational','W_pri_2_b')],
                   tparams[_p('variational','W_pri_mu')],
                   tparams[_p('variational','W_pri_mu_b')],
                   tparams[_p('variational','W_pri_sigma')],
                   tparams[_p('variational','W_pri_sigma_b')],
                   tparams[_p('variational','W_post_1')],
                   tparams[_p('variational','W_post_1_b')],
                   tparams[_p('variational','W_post_2')],
                   tparams[_p('variational','W_post_2_b')],
                   tparams[_p('variational','W_post_mu')],
                   tparams[_p('variational','W_post_mu_b')],
                   tparams[_p('variational','W_post_sigma')],
                   tparams[_p('variational','W_post_sigma_b')],
                   tparams[_p('variational','W_z_1')],
                   tparams[_p('variational','W_z_2')],
                   tparams[_p('variational','W_dec_1')],
                   tparams[_p('variational','W_dec_1_b')],
                   tparams[_p('variational','W_dec_2')],
                   tparams[_p('variational','W_dec_2_b')],
                   tparams[_p('variational','W_dec_mu')],
                   tparams[_p('variational','W_dec_mu_b')],
                   pi,nn]
    if not one_step:
        rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=False,
                                    strict=True)
    else:
        seqs = [mask, state_below_, state_belowx]
        _step = _step_slice_sample
        rval = _step(*(seqs + [init_state, prev_x] + shared_vars))
        return rval[0],rval[1]
    hs = rval[0]
    kl_cost = tensor.sum(rval[1])
    g_xs = rval[2]
    return kl_cost, g_xs


def init_params_vrnn(options):
    params = OrderedDict()

    #embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])
    
    #init_state
    params = get_layer('ff')[0](options, params, prefix='ff_init_state', nin=options['dim_pi'], nout=options['dim'])

    params = get_layer('variational')[0](options, params, prefix='variational',nin=options['dim_word'],dim=options['dim'],dimv=options['dimv'])

    params = get_layer('image_attention')[0](options, params, nin_pi=options['dim_pi'], nin_h=options['dim'], nout=options['dim_pic'])

    params = get_layer('variational_gru')[0](options, params, prefix='variational_gru', nin=options['dim_word'], dim=options['dim'], dimv=options['dimv'])

    return params
    
def build_model(tparams, options, training=True):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    pi = tensor.matrix('pi', dtype='float32')
    pi3 = pi.reshape((pi.shape[0],196, 512))

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    x_flat = x.flatten()
    emb = tparams['Wemb'][x_flat]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])

    kl_cost, g_xs = get_layer('variational_gru')[1](tparams, [pi3, emb], options, mask=x_mask)

    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words_src'] + x_flat
    #cost

    cost = -tensor.log(g_xs.flatten()[x_flat_idx])
    cost = cost.reshape([x.shape[0], x.shape[1]])
    cost = (cost * x_mask).sum(0)
    if not training:
        return trng,use_noise,x,x_mask,pi,opt_ret,cost
    return trng,use_noise,x,x_mask,pi,opt_ret,cost,kl_cost,g_xs
 

#build sampler for captioning
def build_sampler(tparams, options, trng, use_noise):
    sample_x = tensor.vector('sample_x', dtype='int64')
    sample_pi = tensor.matrix('sample_pi', dtype='float32')
    n_timesteps = sample_x.shape[0]
    n_samples = sample_x.shape[1]

    emb = tensor.switch(sample_x[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
                        tparams['Wemb'][sample_x])
    
    mean_pi = tensor.mean(sample_pi,axis=0)
    ini_state = get_layer('ff')[1](tparams, mean_pi, options, prefix='ff_init_state', activ='tanh')
    init_inps = [sample_pi]
    init_out = ini_state
    f_init = theano.function(init_inps, init_out, name='f_init', profile=profile)

    init_state = tensor.matrix('init_state', dtype='float32')
    h, g_x_prob = get_layer('variational_gru')[1](tparams, [sample_pi, emb], options, one_step=True, init_state=init_state, prev_x=emb)

    next_sample = trng.multinomial(pvals=g_x_prob).argmax(1)

    print 'Building f_next...'
    inps = [sample_x, sample_pi,init_state]
    out = [g_x_prob, next_sample, h]
    f_next = theano.function(inps, out, name='f_next', profile=profile)
    print 'Done'

    return f_init,f_next


# generate sample, either with stochastic sampling or beam search. Note that,c
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, pi, options, trng=None, k=1, maxlen=30,stochastic=True, argmax=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'
    pi = numpy.array(pi, dtype='float32')
    pi = pi.reshape((196,512))
    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of vrnn
    next_state = f_init(pi)
    next_w = -1 * numpy.ones((1,)).astype('int64')
    next_state = next_state.reshape((1,next_state.shape[0]))
    for ii in xrange(maxlen):
        pis = numpy.tile(pi, [live_k, 1])
        inps = [next_w, pis, next_state]

        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score -= numpy.log(next_p[0, nw])
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]
            #next_state = next_state.reshape((1,next_state.shape[0]))
            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)
            print next_state.shape

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []

    n_done = 0

    for x, pi in iterator:
        n_done += len(x)

        x, x_mask, pi = prepare_data(x, images = pi, n_words_src=options['n_words_src'])

        pprobs = f_log_probs(x, x_mask, pi)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost, kl_cost, g_xs, x, beta1=0.9, beta2=0.999, e=1e-8):

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, [cost,kl_cost], updates=gsup, profile=profile)

    updates = []

    t_prev = theano.shared(numpy.float32(0.))
    t = t_prev + 1.
    lr_t = lr * tensor.sqrt(1. - beta2**t) / (1. - beta1**t)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')
        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g**2
        step = lr_t * m_t / (tensor.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost, kl_cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, [cost,kl_cost], updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, x, mask, y, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          dimv=100,
          dim_pi=512,
          dim_pic=256,
          dim_l=500,
          encoder='variational_gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_src=100000,  # source vocabulary size
          n_words=100000,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='adadelta',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
          valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'],
          dictionaries=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
          use_dropout=False,
          reload_=False,
          overwrite=False,
          fine_tuning=False,
          fine_tuning_load=""):

    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = pkl.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'Reloading model options'
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    if fine_tuning and os.path.exists(fine_tuning_load):
        print 'Reloading model options'
        with open('%s.pkl' % fine_tuning_load, 'rb') as f:
            model_options = pkl.load(f)
            model_options["dim_pi"]=dim_pi
            model_options["dim_pic"]=dim_pic
            model_options["dimv"]=dimv

    print 'Loading data'
    train = TextIterator(datasets[0], datasets[1],
                         dictionaries[0],
                         n_words_source=n_words_src,
                         batch_size=batch_size,
                         maxlen=maxlen)
    valid = TextIterator(valid_datasets[0], valid_datasets[1],
                         dictionaries[0], 
                         n_words_source=n_words_src, 
                         batch_size=valid_batch_size,
                         maxlen=maxlen)

    print 'Building model'
    params = init_params_vrnn(model_options)
    for k in params.keys():
        print k
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print 'Reloading model parameters'
        params = load_params(saveto, params)

    if fine_tuning and os.path.exists(fine_tuning_load):
        print 'Reloading model parameters'
        params = load_params(fine_tuning_load, params)

    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, pi,\
        opt_ret, \
        cost, kl_cost, g_xs = \
        build_model(tparams, model_options)

    inps = [x, x_mask, pi]

    val_trng, val_use_noise, \
        val_x, val_x_mask, val_pi,\
        val_opt_ret, \
        val_cost = \
        build_model(tparams, model_options,training=False)
    val_inps = [val_x, val_x_mask, val_pi]

    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng, use_noise)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile, on_unused_input='ignore')
    print 'Done'
    
    #f_log_probs for validation
    print 'BUilding f_log_probs for validation...',
    val_f_log_probs = theano.function(val_inps, val_cost, profile=profile)
    
    cost = cost.mean()
    
    cost += kl_cost
    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost, kl_cost, g_xs, x)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, pi in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, pi = prepare_data(x, images=pi, maxlen=maxlen,
                                                n_words_src=n_words_src)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost, kl_cost = f_grad_shared(x, x_mask, pi)
            """print 'Truth'
            for ww in xs[:,5]:
                if ww == 0:
                    break
                if ww in worddicts_r[0]:
                    print worddicts_r[0][ww],
                else:
                    print 'UNK',
            print 
            print 'Sample'
            for ww in g_xs[:,5]:
                w_idx = numpy.argmax(ww)
                if w_idx == 0:
                    break
                if w_idx in worddicts_r[0]:
                    print worddicts_r[0][w_idx],
                else:
                    print 'UNK',
            print
            """
            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud, 'kl-cost ', kl_cost, 'likelihood', cost-kl_cost

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving the best model...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, history_errs=history_errs,
                                uidx=uidx, **unzip(tparams))
                    print 'Done'


            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stochastic = True
                    sample, score = gen_sample(tparams, f_init, f_next,
                                            x[:, jj][:, None],  pi[jj],
                                               model_options, trng=trng, k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=False)
                    print 'Source ', jj, ': ',
                    for vv in x[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[0]:
                            print worddicts_r[0][vv],
                        else:
                            print 'UNK',
                    print
                    """print 'Truth ', jj, ' : ',
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            print worddicts_r[1][vv],
                        else:
                            print 'UNK',
                    print
                    """
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[0]:
                            print worddicts_r[0][vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                val_use_noise.set_value(0.)
                valid_errs = pred_probs(val_f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print 'Valid ', valid_err

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    val_use_noise.set_value(0.)
    valid_err = pred_probs(val_f_log_probs, prepare_data,
                           model_options, valid).mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    return valid_err


if __name__ == '__main__':
    pass
