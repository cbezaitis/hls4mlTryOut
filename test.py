import hls4ml
import qonnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util import cleanup 
from qonnx.util.cleanup import cleanup_model
from torchvision.datasets import CIFAR10

model = ModelWrapper("pot1_no_normalization_on_input.onnx")


model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(RemoveStaticGraphInputs())
model = model.transform(InferDataTypes())
model = cleanup_model(model)

model.save("updated.onnx")

print("Model successfully preprocessed and saved as updated.onnx")


config = hls4ml.utils.config_from_onnx_model(model, granularity='name', default_precision='uint<8>', backend='Catapult')
print("HLS4ML configuration generated from ONNX model:")

print("\n\n\n\n Configuration:")
print(config)
print("\n\n\n\n")

# Manually modify precision for specific layers
config['LayerName']['global_in']['Precision']['result'] = 'uint<8>'   




for layer in config['LayerName']:
    print(f"Layer: {layer}, Config: {config['LayerName'][layer]}")


hls_model = hls4ml.converters.convert_from_onnx_model(
    hls_config=config,
    model='updated.onnx',
    output_dir='./hls_project',
    backend='Catapult',
    io_type='io_parallel'
)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file="precision.jpeg")

hls_model.compile()

builder = CIFAR10
images = builder(root='./input', train=False, download=True)

input_images = []
for i in range(10):
    img, label = images[i]
    img = img.unsqueeze(0)  # Add batch dimension
    input_images.append((img.numpy(), label))
    model.predict(img.numpy())