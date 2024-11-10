import sys
import torch.nn as nn
from cnn_model import Model
from model_handler import ModelHandler
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Following Device will be used for training & testing : " + device.type)
    
    model_file = "model.pth"
    img_res = (200, 200)
    model = Model(img_res, outputClasses=36)
    model_handler = ModelHandler(model, img_res, model_file, device)
    
    arg = sys.argv[1];

    if arg == "train":
        model_handler.train(total_epoch=1)
    elif arg == "test":
        model_handler.test()
    elif arg == "export":
        model_handler.export(filename="model.onnx")

if __name__ == "__main__":
    main()