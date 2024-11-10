import os
import torch
import torch.nn as nn
from cnn_model import Model
from util import prepare_dataloader

class ModelHandler:

    def __init__(self, model: Model, img_res: tuple[int, int], model_file: str, device: torch.device):
        self.model = model
        self.img_res = img_res
        self.model_file = model_file
        self.device = device
        self.batch_size = 32
        if os.path.exists(model_file):
            self.model.load_state_dict(torch.load(model_file))

    def train(self, total_epoch: int):
        self.model = self.model.to(self.device)
        dataloader = prepare_dataloader("dataset/training/", self.img_res, self.batch_size)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, weight_decay = 0.005, momentum = 0.9)  
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(total_epoch):
            for images, labels in dataloader: 
                images = images.to(self.device)
                labels = labels.to(self.device)
                prediction = self.model(images)
                loss = loss_func(prediction, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, total_epoch, loss.item()))      
        torch.save(self.model.state_dict(), self.model_file)

    def test(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        dataloader = prepare_dataloader("dataset/testing/", self.img_res, self.batch_size)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print('Accuracy of the model on the {} images: {} %'.format(total, 100 * correct / total))

    def export(self, filename: str):
        self.model.eval()
        input_format = torch.rand(1, 3, 200, 200)
        onnx_model = torch.onnx.dynamo_export(self.model, input_format)
        onnx_model.save(filename)

