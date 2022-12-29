import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class MSCocoDataset(Dataset):
    """ The dataset class for the MS-COCO captioning dataset. """
    def __init__(self, dataset_dirpath: str, transform, verbose=False):
        """ Initializes the dataset. """
        self.image_captions = dict()

        with open(os.path(dataset_dirpath + '/annotations/captions_train2014.json')) as captions_file:
            json_obj = json.load(captions_file)
            for image_annotation in json_obj.annotations:
                image_id = image_annotation.image_id
                caption = image_annotation.caption

                self.image_captions[image_id] = self.image_captions.get(image_id, []).append(caption)
        if verbose:
            print("Loaded captions for {} images.".format(len(self.image_captions)))

        self.dirpath = dataset_dirpath
        self.transform = transform

    def __len__(self):
        return len(self.image_captions)

    def __getitem__(self, item):
        """ Loads the image in memory. Check if the image is downloaded, if not downloads it."""


class Flickr8kDataset(Dataset):
    """ The dataset class for the Flickr 8k captioning dataset. """

    def __init__(self):

    def __len__(self):

    def __getitem__(self, item):
