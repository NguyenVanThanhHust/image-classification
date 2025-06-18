from torchvision.datasets import ImageFolder

if __name__ == '__main__':
    dataset = ImageFolder("../Datasets/split_mini_imagenet")
    print(dataset.__len__())