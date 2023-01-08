import argparse
import os

import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt

import models
import datasets
import metrics
import inference
from query_embedder import QueryEmbedder, load_glove_vocabulary
from model_trainer import train_model


def main_training(args):
    """ Entry point for the training task. """
    # Load the query embedder.
    qe = QueryEmbedder()

    # Load the dataset.
    dataset = datasets.load_dataset(args.dataset, args.dataset_dirpath, qe)

    # Load the model or create a new one.
    model = models.get_class(args.model)(qe)
    if args.load_filepath:
        model.load_state_dict(torch.load(args.load_filepath))

    # Train.
    training_statistics = train_model(model, dataset, args)

    # Output.

    # Save the model if required.
    if args.save_filepath:
        torch.save(model.state_dict(), args.save_filepath)


def main_inference(
        args,
        load_model_fn, # Function that takes the name of the model + the saved model filepath + the QE
        get_captions_fn, # Function that takes the paths to the images and
    ):
    """ Entry point for the inference task. """
    # Load the query embedder.
    print("Loading the query embedder...")
    qe = QueryEmbedder()

    print("Loading the GloVe vocabulary...")
    vocabulary = load_glove_vocabulary("pretrained/glove.6B.50d.txt")

    # Load the database.
    print("Loading the image database...")
    image_filepaths = [
        os.path.join(args.dataset_dirpath, x)
        for x in os.listdir(args.dataset_dirpath)
        if os.path.isfile(os.path.join(args.dataset_dirpath, x))# and os.path.splitext(x)[-1] in ["png", "jpg", "jpeg"]
    ]
    print("\tLoaded {} images.".format(len(image_filepaths)))

    # Load the model.
    print("Loading the model...")
    model, device = load_model_fn(args.model, args.load_filepath, qe)

    # Compute the captions for all the images.
    print("Computing the captions for the images...")
    captions = get_captions_fn(model, device, image_filepaths, vocabulary)
    print(list(zip(image_filepaths, captions)))

    # Compute the score between the captions and the query.
    print("Computing the scores between the captions and the query...")
    scores = inference.compute_query_scores(args.metric, args.query, captions)
    print("Scores:", scores)

    # Fetch back the best image.
    print("Fetching back the best image...")
    best_matching_image_filepath = image_filepaths[np.argmax(scores)]
    best_matching_image = PIL.Image.open(best_matching_image_filepath)

    print("Best matching image: {}".format(best_matching_image_filepath))
    print("\tWith score: {}".format(np.max(scores)))
    plt.imshow(best_matching_image)
    plt.show()


def parse_arguments() -> dict:
    """ Parses the command-line arguments. """
    parser = argparse.ArgumentParser(
        prog='MVA-DL-Project-2022',
        description='This program allows training and doing inference for captioning-based image retrieval models.'
                    ' Models available: CNN-LSTM baseline,  Semantic Composition Network (Gan et al., 2017),'
                    ' AttentiveAttention (Lu et al., 2017).'
    )

    # Task arguments.
    parser.add_argument('-t', '--task', choices=['inference', 'training'], default='inference',
                        help='Sets the task for the program.')
    parser.add_argument('-o', '--output_filepath', default='output.txt', help='If set, the filepath where to save the output.')

    # For inference mode only.
    parser.add_argument('-q', '--query', type=str, help='[INFERENCE ONLY] The query to look evaluate.')

    # Dataset arguments.
    parser.add_argument('-d', '--dataset', type=str, choices=datasets.AVAILABLE_DATASETS, default='MS-COCO',
                        help='Chooses which dataset to use.')
    parser.add_argument('-dp', '--dataset_dirpath', type=str, help='Sets the directory path to the dataset.')
    parser.add_argument('-p', '--percentage', type=float, help='Amount of the dataset to use.')

    # Model arguments.
    parser.add_argument('-m', '--model', choices=models.AVAILABLE_MODELS, default='baseline',
                        help='Sets the model to use.')
    parser.add_argument('-l', '--load_filepath', type=str, help='If set, the filepath to the model to load.')
    parser.add_argument('-s', '--save_filepath', type=str, help='If set, the filepath where to save the model.')

    # Train.
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train for.')

    # Inference.
    parser.add_argument('-mt', '--metric', choices=metrics.AVAILABLE_METRICS, default='BLEU',
                        help='Sets the metric to use.')

    # Other arguments.
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='If set, the program will output more information.')

    return parser.parse_args()


def main():
    """ Parses the args and delegates the task work. """
    args = parse_arguments()

    # Check we have all the information before starting.
    # Inference mode.
    if args.task == 'inference':
        if not args.query:
            print('Inference mode requires a query to evaluate.')
            return

        if not args.dataset_dirpath:
            print('Inference mode requires a database to evaluate the query against.')
            return
        else:
            if not os.path.isdir(args.dataset_dirpath):
                print('The dataset dirpath doesn\'t exist or is not accessible.')
                return

        if not args.load_filepath:
            print('Inference mode requires a model to be loaded.')
            return
        else:
            if not os.path.isfile(args.load_filepath):
                print('The model filepath doesn\'t exist or is not accessible.')
                return

        # In the case of a PyTorch model.
        main_inference(args, inference.load_pytorch_model, inference.get_captions_pytorch_model)
    # Training mode.
    else:
        if (not args.dataset) or (not args.dataset_dirpath):
            print('Training mode requires a dataset to be used as the database to evaluate the query against.')
            return
        else:
            if not os.path.isdir(args.dataset_dirpath):
                print('The dataset dirpath doesn\'t exist or is not accessible.')
                return

        if args.load_filepath:
            if not os.path.isfile(args.load_filepath):
                print('The model filepath doesn\'t exist or is not accessible.')
                return

        main_training(args)


if __name__ == '__main__':
    main()
