import argparse
import os

import models
import datasets
from query_embedder import QueryEmbedder
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
        model.load(args.load_filepath)

    # Train.
    training_statistics = train_model(model, dataset, args)

    # Output.

    # Save the model if required.
    if args.save_filepath:
        model.save(args.save_filepath)


def main_inference(args):
    """ Entry point for the inference task. """
    # Load the dataset.
    dataset = datasets.load_dataset(args.dataset, args.dataset_dirpath)

    # Load the model.
    model = models.get_class(args.load_filepath)()
    model.load(args.load_filepath)

    # Prepare the query.
    qe = QueryEmbedder()
    embedded_query = qe.encode(args.query)

    # Evaluate the query.
    prediction = model.predict(embedded_query, dataset)

    # Output.



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

        if (not args.dataset) or (not args.dataset_dirpath):
            print('Inference mode requires a dataset to be used as the database to evaluate the query against.')
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

        main_inference(args)
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
