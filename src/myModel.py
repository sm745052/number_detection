import tensorflow as tf
class dnnModel:
    def __init__(self):
        '''
            dnnModel is used to make a sequential model with given input shape and no of output classes.
        '''
        pass
    
    def create(self, input_shape, output_shape):
        '''
            Creates a sequential deep neural network model.
            input_shape: the shape of the input data
            output_shape: no of classes



            returns
            ----------------
            model
            ----------------
        '''
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(output_shape*10, activation = tf.nn.relu)(x)
        x = tf.keras.layers.Dense(output_shape*5, activation = tf.nn.relu)(x)
        x = tf.keras.layers.Dense(output_shape, activation = tf.nn.softmax)(x)
        model = tf.keras.Model(inputs = inputs, outputs = x)
        model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), metrics=['accuracy'])
        return model