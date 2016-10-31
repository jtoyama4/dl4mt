import cPickle as pkl
import gzip
import numpy
import os.path

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,image,global_fc7,class_txt,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        self.class_txt = fopen(class_txt, 'r')
        self.image_list = fopen(image, 'r')
        self.image_basedir = os.path.dirname(image)
        self.global_fc7 = numpy.load(global_fc7)
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.end_of_data = False
        
        self.count = 0

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        self.class_txt.seek(0)
        self.image_list.seek(0)
        self.count = 0

    def get_index(self,cls,top_n):
        with open(os.path.join(self.image_basedir,cls.strip()),'r') as f:
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

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        image = []

        try:
            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]
                
                # read from source file and map to word index
                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    self.count += 1
                    continue
                
                # read image
                # image shape : (num_of_objects, image_feature)
                cls = self.class_txt.readline()
                idx = self.get_index(cls,1)
                fc7_global = self.global_fc7[self.count]
                fc7_global = fc7_global[numpy.newaxis,:]
                ii = numpy.load(os.path.join(self.image_basedir,self.image_list.readline().strip()))[idx]
                ii = numpy.concatenate((fc7_global,ii),axis=0)
                source.append(ss)
                target.append(tt)
                image.append(ii)

                self.count += 1

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target, image
