import cPickle
import numpy as np
import random

image_height = 32
image_width = 32
image_channels = 3
label_number = 10

def unpickle(filename):
    with open(filename, 'rb') as fo:
        pickled = cPickle.load(fo)
    return pickled

class Cifar10Batch:
    def __init__(self, data=None, labels=None):
        self.data = data
        self.labels = labels

class Cifar10Dataset:
    def __init__(self, filename=None, one_hot=True, reshape=False):
        self.data = None
        self.labels = None
        self.one_hot = one_hot
        self.flatten = reshape
        if not filename is None:
            dict_obj = unpickle(filename)
            data = dict_obj['data']
            labels = dict_obj['labels']
            sample_number = len(data)
            data_array = np.zeros([sample_number, image_height*image_width*image_channels]) if self.flatten else np.zeros([sample_number, image_height, image_width, image_channels])
            labels_array = np.zeros([sample_number, label_number]) if self.one_hot else np.zeros([sample_number])
            data_aitem = np.zeros([image_height, image_width, image_channels])
            for i in range(sample_number):
                if self.one_hot:
                    labels_array[i,labels[i]] = 1.0
                else:
                    labels_array[i] = float(labels[i])
                data_tmp = np.reshape(data[i], [image_channels,image_height,image_width])
                data_aitem[:,:,0] = data_tmp[0,:,:]
                data_aitem[:,:,1] = data_tmp[1,:,:]
                data_aitem[:,:,2] = data_tmp[2,:,:]
                if self.flatten:
                    data_array[i,:] = np.reshape(data_aitem,[image_height*image_width*image_channels])
                else:
                    data_array[i,:,:,:] = data_aitem
            self.data = data_array.astype(np.float32)
            self.labels = labels_array.astype(np.float32)
        self._renew()

    def extend(self, dataset):
        if type(dataset) == str:
            self.extend(Cifar10Dataset(filename=dataset, one_hot=self.one_hot, reshape=self.flatten))
        elif isinstance(dataset, Cifar10Dataset) and (dataset.sample_number > 0):
            new_data = dataset.data
            new_labels = dataset.labels
            if self.one_hot and (not dataset.one_hot):
                new_new_labels = np.zeros([dataset.sample_number, label_number])
                new_new_labels[:,new_labels[i]] = 1.0
                new_labels = new_new_labels
            elif (not self.one_hot) and dataset.one_hot:
                new_new_labels = np.where(new_labels[:,:]==1)[1]
                if new_new_labels.shape[0] != dataset.sample_number:
                    raise Exception("Error when parsing one-hot labels: not enough labels")
                new_labels = new_new_labels
            if self.flatten and (not dataset.flatten):
                new_data = np.reshape(new_data, [dataset.sample_number, image_height*image_width*image_channels])
            elif (not self.flatten) and dataset.flatten:
                new_data = np.reshape(new_data, [dataset.sample_number, image_height, image_width, image_channels])
            self.data = new_data if (self.data is None) else np.append(self.data, new_data, axis=0)
            self.labels = new_labels if (self.labels is None) else np.append(self.labels, new_labels, axis=0)
            self._renew()

    def _renew(self):
        self._data = None
        self._labels = None
        self.sample_number = 0 if (self.data is None) else len(self.data)
        # to force refresh next time
        self._epoches_completed = -1
        self._index_in_epoch = self.sample_number + 1

    def next_batch(self, batch_size):
        if self._index_in_epoch + batch_size > self.sample_number:
            lst = range(self.sample_number)
            random.shuffle(lst)
            self._data = self.data[lst,:] if self.flatten else self.data[lst,:,:,:]
            self._labels = self.labels[lst,:] if self.one_hot else self.labels[lst]
            self._epoches_completed += 1
            self._index_in_epoch = 0
        start = self._index_in_epoch
        end = self._index_in_epoch + batch_size
        self._index_in_epoch = end
        batch_data = self._data[start:end,:] if self.flatten else self._data[start:end,:,:,:]
        batch_labels = self._labels[start:end,:] if self.one_hot else self._labels[start:end]
        return Cifar10Batch(data=batch_data, labels=batch_labels)

class Cifar10:
    def __init__(self, dirname):
        self.train = Cifar10Dataset()
        for i in range(1,6):
            filename = "data_batch_%d" % i
            self.train.extend(dirname + "/" + filename)
            print "[CIFAR-10 Dataset] Loaded", filename
        print "[CIFAR-10 Dataset] Training set: %d" % self.train.sample_number
        filename = "test_batch"
        self.test = Cifar10Dataset(dirname + "/" + filename)
        print "[CIFAR-10 Dataset] Loaded", filename
        print "[CIFAR-10 Dataset] Testing set: %d" % self.test.sample_number
        filename = "batches.meta"
        self.names = unpickle(dirname + "/" + filename)['label_names']
        print "[CIFAR-10 Dataset] Loaded", filename

dataset = Cifar10("../cifar-10-batches-py")
