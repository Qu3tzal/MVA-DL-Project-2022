import PIL.Image
from torchvision.transforms import Compose, PILToTensor, ToTensor, Resize
from torchvision.datasets import CocoCaptions, VisionDataset
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


class CustomFlickr8kDataset(VisionDataset):
    def __init__(self,
                 root,
                 annotations_file_filepath,
                 transform=None,
                 target_transform=None) -> None:
        if root[-1] != '/':
            root = root + '/'
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        self.ids = [

        ]
        self.targets = dict()

        with open(annotations_file_filepath, 'rt') as annotations_file:
            lines = annotations_file.readlines()

            for line in lines:
                name_id = line.split("#")[0]
                target = line.split("#")[1]
                self.ids.append(name_id)
                self.targets[name_id] = self.targets.get(name_id, [target])

    def __getitem__(self, index: int):
        img = PIL.Image.open(self.root + self.ids[index])
        targets = self.targets[self.ids[index]]

        return self.transform(img), self.target_transform(targets)

    def __len__(self) -> int:
        return len(self.ids)


def get_flickr8k_dataset(dirpath, embedder):
    def target_transform_fn(targets):
        # TODO: use all the captions, not just one.
        encoded_caption = embedder.encode(targets[0])
        return encoded_caption

    ds = CustomFlickr8kDataset(root=dirpath + '/images',
               annotations_file_filepath=dirpath + '/captions.txt',
               target_transform=target_transform_fn,
               transform=Compose([
                   Resize((224, 224)),
                   ToTensor()
               ])
            )
    return ds
