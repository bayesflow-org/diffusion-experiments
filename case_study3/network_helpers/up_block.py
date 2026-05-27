import keras
from typing import Any
import bayesflow as bf
from bayesflow.utils.serialization import deserialize, serialize
from bayesflow.utils import sequential_kwargs
from bayesflow.utils.tensor_utils import Tensor


@bf.utils.serialization.serializable("custom")
class UpSample(keras.Layer):
    def __init__(self,
                 out_channels: int,
                 interpolation: str = "nearest",
                 kernel_init: str = None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.interpolation = interpolation
        self.kernel_init = kernel_init
        self.upsample_op = keras.Sequential([
            keras.layers.UpSampling2D(size=2, interpolation=interpolation),
            keras.layers.Conv2D(out_channels, kernel_size=1, padding="same", kernel_initializer=kernel_init),
        ])

    def call(self, inputs: Tensor, training: bool = None, **kwargs) -> Tensor:
        x = self.upsample_op(inputs, training=training)
        return x

    def build(self, input_shape):
        if self.built:
            return
        self.upsample_op.build(input_shape)

    def compute_output_shape(self, input_shape):
        up_shape = self.upsample_op.compute_output_shape(input_shape)
        return up_shape

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = sequential_kwargs(base_config)
        config = {
            "out_channels": self.out_channels,
            "interpolation": self.interpolation,
            "kernel_init": self.kernel_init,
        }
        return base_config | serialize(config)

if __name__ == "__main__":
    spatial = keras.Input(shape=(256, 256, 64), name="spatial_input")
    spatial_down = UpSample(out_channels=32, name="spatial_down")
    out = spatial_down(spatial)
    model = keras.Model(inputs=spatial, outputs=out)
    model.summary()
    print("Before calling the model:")
    for layer in model.layers:
        print(f"Layer '{layer.name}': built = {layer.built}")
    print(out)