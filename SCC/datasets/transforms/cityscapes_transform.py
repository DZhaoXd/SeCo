from torchvision import transforms


def cityscapes_train(mean, std):
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        # transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


def cityscapes_test(mean, std):
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform