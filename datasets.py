from torchvision.transforms import PILToTensor
from torchvision.datasets import CocoCaptions, Flickr8k


AVAILABLE_DATASETS = ['MS-COCO', 'Flickr-8k']


def load_dataset(dataset_name, dirpath):
    if dataset_name == "MS-COCO":
        return get_ms_coco_dataset(dirpath)
    elif dataset_name == "Flickr-8k"
        return get_flickr8k_dataset(dirpath)

def get_ms_coco_dataset(dirpath):
    ds = CocoCaptions(root=dirpath + '/images',
                    annFile=dirpath + '/annotations.txt',
                    transform=PILToTensor())
    return ds


def get_flickr8k_dataset(dirpath):
    ds = Flickr8k(root=dirpath + '/images',
               annFile=dirpath + '/annotations.txt',
               transform=PILToTensor())
    return ds
