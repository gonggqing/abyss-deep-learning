import keras.zoo.segmentation.deeplab_v3_plus.model

def make_model( config ): # where config specifies input_shape, ..., regularization?
    # todo
    if isinstance( config, dict ):
        pass
    if isinstance( config, str ):
        # todo: load config from file
        config = json.read( config )
    else:
        # throw
    m = make_model( config[ 'input_shape' ], ... )
    pass

def make_model( input_shape = ..., ... ):
    # todo
    m.regularize( config[...] )
    pass

def load( model_definition, model_weights ): # should it be totally generic and thus be in keras.models.py?
    # todo
    pass

def save( model, model_definition, model_weights ): # should it be totally generic and thus be in keras.models.py?
    # todo
    pass
