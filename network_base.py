import abc
import sys

import tensorflow as tf

DEFAULT_PADDING = 'SAME'

def layer(op):
    """ Decorator for composable network layers.
    """

    def layer_decorated(self, *args, **kwargs):
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        
        layer_output = op(self, layer_input, *args, **kwargs)
        
        self.layers[name]= layer_output
        self.feed(layer_output)
        return self
    
    return layer_decorated


class BaseNetwork(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = inputs
        self.terminals = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    @abc.abstractmethod
    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')
        
    def feed(self, *args):
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            try:
                is_str = isinstance(fed_layer, basestring)
            except NameError:
                is_str = isinstance(fed_layer, str)
            if is_str:
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self
    
    

    def make_var(self, name, shape, trainable=True):
        return tf.get_variable(name, shape, trainable=trainable)

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_i,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             trainable=True,
             biased=True):
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
            output = convolve(input, kernel)

            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            
            return output

    @layer
    def batch_normalization(self, input, num_feature, name, scale_offset=True, relu=False):
        with tf.variable_scope(name) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input,
                mean=self.make_var('mean', shape=shape),
                variance=self.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output

    @layer
    def concat(self, input, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)
    
    @layer
    def expand_dim(self, input, axis, name):
        return tf.expand_dims(input, axis=axis, name=name)

    @layer
    def squeeze(self, input, axis, name):
        return tf.squeeze(input, axis=axis, name=name)
        
    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)
    

    @layer
    def tanh(self, input, name):
        return tf.nn.tanh(input, name=name)
    
    # Math
    @layer
    def sum(self, x, y, name):
        return tf.math.reduce_sum(x, y, name=name)

    @layer
    def pow(self, x, y, name):
        return tf.math.pow(x, y, name=name)
    
    @layer
    def div(self, x, y, name):
        return tf.math.divide(x, y, name=name)

    