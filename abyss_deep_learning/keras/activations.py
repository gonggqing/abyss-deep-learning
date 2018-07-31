from keras.engine.base_layer import InputSpec
from keras.layers import Layer
import keras.backend as K
import keras.constraints
import keras.initializers
import keras.regularizers

#### Layers ####
## Activation functions are found as static methods of the layers called _activation().

class Hexpo(Layer):
    """
    Hexpo activation layer.
   https://ieeexplore-ieee-org.ezproxy1.library.usyd.edu.au/stamp/stamp.jsp?tp=&arnumber=7966168
   """
    @staticmethod
    def _activation(x, a, b, c, d):
        """Hexpo activation function, returns tensor."""
        shape = K.tf.shape(x)
        piece1 = -a * (K.tf.exp(- x / b) - 1)
        piece2 = c * (K.tf.exp(x / d) - 1)
        res = K.tf.where(
            K.tf.greater_equal(x, 0),
            x=piece1,
            y=piece2)
        
#         x = K.tf.reshape(x, [-1])
        return res #K.tf.reshape(y, shape)

    def __init__(
                self, layer_depth=0,
                initializers=None, regularizers=None, constraints=None,
                shared_axes=None, **kwargs):
        '''layer_depth should be set to the depth that this layer will be.
        initializers, regularizers and constraints can be either a keras string to be duplicated
        amongst the a, b, c, d parameters, or a dict mapping.
        
        Example:
            params = {
                'constraints': {'a': None, 'b': None, 'c': None, 'd': None},
                'initializers': {
                    'a': {'class_name': 'Constant', 'config': {'value': 1.0}},
                    'b': {'class_name': 'Constant', 'config': {'value': 1.0}},
                    'c': {'class_name': 'Constant', 'config': {'value': 1.0}},
                    'd': {'class_name': 'Constant', 'config': {'value': 1.0}}},
                'regularizers': {'a': None, 'b': None, 'c': None, 'd': None}
                }
            Hexpo(layer_depth=0, shared_axes=None, **params)
            '''
        super().__init__(**kwargs)

        self.supports_masking = True
        self.param_names = ['a', 'b', 'c', 'd']
        self.params = None
        self.initializers = {key: 
            keras.initializers.Constant(value=(1 + layer_depth / 2)) for key in self.param_names}
        self.regularizers = {key: None for key in self.param_names}
        self.constraints = {key: None for key in self.param_names}

        if isinstance(initializers, str):
            self.initializers = {key: initializers.get(initializers) for key in self.param_names}
        elif isinstance(initializers, dict):
            self.initializers = initializers
        elif initializers is not None:
            raise ValueError("Hexpo given bad initializer string/list")

        if isinstance(regularizers, str):
            self.regularizers = {key: regularizers.get(regularizers) for key in self.param_names}
        elif isinstance(regularizers, dict):
            self.regularizers = regularizers
        elif regularizers is not None:
            raise ValueError("Hexpo given bad regularizers string/list")

        if isinstance(constraints, str):
            self.constraints = {key: constraints.get(constraints) for key in self.param_names}
        elif isinstance(initializers, dict):
            self.constraints = constraints
        elif constraints is not None:
            raise ValueError("Hexpo given bad constraints string/list")

        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True
        self.params = {
            param_name: self.add_weight(
                shape=param_shape,
                name=param_name,
                initializer=self.initializers[param_name],
                regularizer=self.regularizers[param_name],
                constraint=self.constraints[param_name])
            for param_name in self.param_names}
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs, mask=None):
        return Hexpo._activation(inputs, **self.params)

    def get_config(self):
        def items_sorted(a_dict):
            return dict(sorted(a_dict.items(), key=lambda k: k[0])).items()
        config = {
            'initializers': {param_name: keras.initializers.serialize(initializer)
                for param_name, initializer in items_sorted(self.initializers)},
            'regularizers': {param_name: keras.regularizers.serialize(regularizer)
                for param_name, regularizer in items_sorted(self.regularizers)},
            'constraints': {param_name: keras.constraints.serialize(constraint)
                for param_name, constraint in items_sorted(self.constraints)},
            'shared_axes': self.shared_axes
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape