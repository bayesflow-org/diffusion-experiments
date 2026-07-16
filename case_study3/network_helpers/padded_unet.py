import keras
import bayesflow as bf
from bayesflow.utils.serialization import deserialize, serialize
from bayesflow.utils import sequential_kwargs

from .residual import Residual
from .double_conv import DoubleConv
from .sequential import Sequential

from .down_block import DownSample
from .up_block import UpSample


@bf.utils.serialization.serializable("custom")
class PaddedUNetSubnet(keras.Layer):
    def __init__(
            self,
            widths=(8, 16, 32),
            activation="mish",
            use_batchnorm=False,
            dropout=0.0,
            num_res_blocks=(1, 1, 1),
            pad_size=3,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.widths = widths
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.num_res_blocks = num_res_blocks
        self.stages_down = []
        self.stages_downsample = []
        self.stages_up = []
        self.stages_upsample = []
        self.pad = pad_size
        for s, (num_res, width) in enumerate(zip(num_res_blocks, widths)):
            layers = []
            layers.append(keras.layers.Lambda(
                lambda x: keras.ops.pad(x, pad_width=((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="reflect"),
                output_shape=lambda x: (x[0], x[1] + 6, x[2] + 6, x[3]),
            ))
            for r in range(num_res):
                layer = DoubleConv(width, use_batchnorm=use_batchnorm, dropout=dropout, activation=activation)
                layer = Residual(layer)
                act = keras.activations.get(activation)
                if not isinstance(act, keras.Layer):
                    act = keras.layers.Activation(act)
                layers.append(layer)
                layers.append(act)
            layers.append(keras.layers.Cropping2D(cropping=self.pad))
            if s < len(widths) - 1:
                self.stages_downsample.append(DownSample(out_channels=widths[s+1]))
            self.stages_down.append(Sequential(layers))

        for s, (num_res, width) in enumerate(zip(reversed(num_res_blocks), reversed(widths))):
            layers = []
            if s < len(widths) - 1:
                self.stages_upsample.append(UpSample(out_channels=widths[s+1]))
            if s > 0:
                layers.append(keras.layers.Lambda(
                    lambda x: keras.ops.pad(x, pad_width=((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="reflect"),
                    output_shape=lambda x: (x[0], x[1] + 6, x[2] + 6, x[3]),
                ))
                for r in range(num_res):
                    layer = DoubleConv(width, use_batchnorm=use_batchnorm, dropout=dropout, activation=activation)
                    layer = Residual(layer)
                    act = keras.activations.get(activation)
                    if not isinstance(act, keras.Layer):
                        act = keras.layers.Activation(act)
                    layers.append(layer)
                    layers.append(act)
                layers.append(keras.layers.Cropping2D(cropping=self.pad))
                self.stages_up.append(Sequential(layers))

        self.last_channels = widths[-1]

    #def call(self, x, t, conditions, training=False):
    def call(self, inputs, training=False):
        x, t, conditions = inputs
        t = keras.ops.broadcast_to(t, keras.ops.shape(x)[:-1] + (1,))
        h = keras.ops.concatenate((x, t, conditions), axis=-1)
        skip_connections = []
        for s in range(len(self.widths)):
            h = self.stages_down[s](h, training=training)
            if s < len(self.widths) - 1:
                skip_connections.append(h)
                h = self.stages_downsample[s](h, training=training)
        skip_connections = list(reversed(skip_connections))
        for s in range(len(self.widths)-1):
            h = self.stages_upsample[s](h, training=training)
            h = keras.ops.concatenate((h, skip_connections[s]), axis=-1)
            h = self.stages_up[s](h, training=training)
        return h

    def build(self, x_shape, t_shape, conditions_shape):
    #def build(self, input_shape):
        #x_shape, t_shape, cond_shape = input_shape
        if self.built:
            return
        x, t, c = keras.ops.zeros(x_shape), keras.ops.zeros(t_shape), keras.ops.zeros(conditions_shape)
        self.call(x, t, c)

    #def compute_output_shape(self, x_shape, t_shape, conditions_shape):
    def compute_output_shape(self, input_shape):
        x_shape, t_shape, cond_shape = input_shape
        x, t, c = keras.ops.zeros(x_shape), keras.ops.zeros(t_shape), keras.ops.zeros(cond_shape)
        out = self.call(x, t, c)
        out_shape = keras.ops.shape(out)
        return out_shape

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = sequential_kwargs(base_config)

        config = {
            "widths": self.widths,
            "activation": self.activation,
            "use_batchnorm": self.use_batchnorm,
            "dropout": self.dropout,
            "num_res_blocks": self.num_res_blocks,
            "pad_size": self.pad,
        }

        return base_config | serialize(config)
