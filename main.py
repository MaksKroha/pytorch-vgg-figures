import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.custom_dataset import MnistDataset
from src.model.neural_net import CNN
from src.evaluate import evaluate_dataloader
from src.train import train
from src.utils.formatter import format_paths_into_csv_name_label

if __name__ == "__main__":
    root_dir = "/home/maksymkroha/MineFiles/kaggle/mnist-digits/trainingSet"
    paths_csv = "dataset/paths.csv"
    trained_model = "models/trained_model.pt"

    # folder_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #
    # for data in os.listdir(root_dir):
    #     if data in folder_names:
    #         format_paths_into_csv_name_label(f"{root_dir}/{data}",
    #                                          paths_csv,
    #                                          data)

    # first of all need to create paths.csv file
    # in which there will be all paths to all images
    # csv format - <path, label>

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28))
    ])

    # probability to be zeroed
    dropout = 0.2
    lr = 0.001
    lr_min = 0.00001
    epochs = 3
    t_max = 3
    device = "cpu"

    model = CNN(dropout)

    try:
        model.load_state_dict(torch.load(trained_model))
    except Exception as e:
        print(e)

    # dataset configuration
    dataset = MnistDataset(paths_csv, transform=transforms)

    # training
    dataloader_train = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    train(model, dataloader_train, epochs, device, lr, t_max, lr_min)

    torch.save(model.state_dict(), trained_model)

    # evaluating
    dataloader_test = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    evaluate_dataloader(model, dataloader_test, 10)
