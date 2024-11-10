from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def prepare_dataloader(path: str, image_res: tuple[int, int], batch_size: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(image_res),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(path, transform)
    return DataLoader(dataset, batch_size, shuffle=True)