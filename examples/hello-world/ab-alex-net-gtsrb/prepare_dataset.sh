DATASET_ROOT="~/data"

python3 -c "import torchvision.datasets as datasets; datasets.GTSRB(root='${DATASET_ROOT}', transform=transforms, download=True, split="train")"
