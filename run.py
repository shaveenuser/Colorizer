import coremltools as ct
import torch
import colorizers

torch_model = colorizers.eccv16().eval()

example_input = torch.rand([1, 1, 256, 256])
traced_model = torch.jit.trace(torch_model, example_input)

coreml_model = ct.convert(traced_model, 
                          inputs=[ct.TensorType(name="input", shape=example_input.shape)])

coreml_model.save("Colorizer.mlpackage")