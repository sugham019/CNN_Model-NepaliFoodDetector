import sys
import onnxruntime as ort
import numpy as np
import cv2
from torchvision import transforms

demo_img = sys.argv[1]

image = cv2.imread(demo_img)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (200, 200))

transform = transforms.Compose([
    transforms.ToTensor()
])

image_tensor = transform(image)
image_numpy = image_tensor.numpy()
image_numpy = np.expand_dims(image_numpy, axis=0)

onnx_model = ort.InferenceSession('model.onnx')
input_name = onnx_model.get_inputs()[0].name

output = onnx_model.run(None, {input_name: image_numpy})
index = np.argmax(output)

print(output)
print("Output Class : ", index)