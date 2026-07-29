from utils.dataloader import get_dataloaders

def main():

    train_loader, val_loader, test_loader = get_dataloaders()

    print("=" * 60)
    print("Train batches :", len(train_loader))
    print("Val batches   :", len(val_loader))
    print("Test batches  :", len(test_loader))

    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)


if __name__ == "__main__":
    main()