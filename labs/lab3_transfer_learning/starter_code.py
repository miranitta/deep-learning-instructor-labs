from torchvision.models import resnet18


def build_model(num_classes=10):
    model = resnet18(weights="DEFAULT")
    # TODO: freeze backbone
    # TODO: replace final layer
    return model
