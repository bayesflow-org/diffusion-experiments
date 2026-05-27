import bayesflow as bf
import keras

from .resnet import ResNet
from .sequential import Sequential


@bf.utils.serialization.serializable("custom")
class ResNetSummary(bf.networks.SummaryNetwork):
    def __init__(
        self,
        widths,
        summary_dim=8,
        activation="mish",
        dropout=0.0,
        use_batchnorm=False,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.summary_dim = summary_dim

        layers = []
        layers.append(ResNet(widths, use_batchnorm=use_batchnorm, activation=activation, dropout=dropout))
        layers.append(keras.layers.Conv2D(filters=summary_dim, kernel_size=1))
        layers.append(keras.layers.GlobalAveragePooling2D())
        self.net = Sequential(layers)

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)
