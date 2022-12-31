from torchvision.transforms import Compose, PILToTensor, ToTensor, Resize
from torchvision.datasets import CocoCaptions, Flickr8k
import query_embedder as qe


AVAILABLE_DATASETS = ['MS-COCO', 'Flickr-8k']


def load_dataset(dataset_name, dirpath, embedder):
    if dataset_name == "MS-COCO":
        return get_ms_coco_dataset(dirpath, embedder)
    elif dataset_name == "Flickr-8k":
        return get_flickr8k_dataset(dirpath, embedder)


def get_ms_coco_dataset(dirpath, embedder):
    def target_transform_fn(targets):
        # TODO: use all the captions, not just one.
        encoded_caption = embedder.encode(targets[0])
        return encoded_caption

    ds = CocoCaptions(root=dirpath + '/images',
                    annFile=dirpath + '/captions.json',
                    target_transform=target_transform_fn,
                    transform=Compose([
                        Resize((224, 224)),
                        ToTensor()
                    ])
                )
    return ds


def get_flickr8k_dataset(dirpath, embedder):
    ds = Flickr8k(root=dirpath + '/images',
               annFile=dirpath + '/captions.json',
               transform=Compose([
                   Resize((224, 224)),
                   ToTensor()
               ])
            )
    return ds
