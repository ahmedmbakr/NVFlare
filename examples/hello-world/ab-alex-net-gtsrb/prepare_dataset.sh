DATASET_ROOT="~/data"

python3 -c "import torchvision.datasets as datasets; datasets.GTSRB(root='${DATASET_ROOT}', download=True, split='train')"
python3 -c "import torchvision.datasets as datasets; datasets.GTSRB(root='${DATASET_ROOT}', download=True, split='test')"
