from pickle import load

import tqdm
from numpy import argmax
from keras.applications.vgg16 import VGG16
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
import json
import random
import matplotlib.pyplot as plt
import string

import metrics


def tokenizer_fn(input_str):
    # Lowercase
    # Remove all punctuation
    # Remove before/after whitespaces
    # Split on whitespaces
    return input_str.lower().translate(str.maketrans('', '', string.punctuation)).strip().split(' ')


"""
CODE FROM NOTEBOOK
<START>
"""


def extract_feature(model, file_name):
    img = load_img(file_name, target_size=(224, 224))  # size is 224,224 by default
    x = img_to_array(img)  # change to np array
    x = np.expand_dims(x, axis=0)  # expand to include batch dim at the beginning
    x = preprocess_input(x)  # make input confirm to VGG16 input format
    fc2_features = model.predict(x)
    return fc2_features


def generate_caption(pred_model, caption_train_tokenizer, photo, max_length):
    in_text = '<START>'
    caption_text = list()
    for i in range(max_length):
        # integer encode input sequence
        sequence = caption_train_tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        model_softMax_output = pred_model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        word_index = argmax(model_softMax_output)
        # map integer to word
        word = caption_train_tokenizer.index_word[word_index]
        # print(word)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word != 'end':
            caption_text.append(word)
        if word == 'end':
            break
    return caption_text


def flatten(lst):
    return sum(([x] if not isinstance(x, list) else flatten(x) for x in lst), [])


def generate_caption_beam(pred_model, caption_train_tokenizer, photo, max_length, vocab_size, beam_width):
    sequence = caption_train_tokenizer.texts_to_sequences(['<START>'])[0]
    sequence = pad_sequences([sequence], maxlen=max_length)
    model_softMax_output = np.squeeze(pred_model.predict([photo, sequence], verbose=0))
    most_likely_seq = np.argsort(model_softMax_output)[-beam_width:]
    most_likely_prob = np.log(model_softMax_output[most_likely_seq])

    most_likely_cap = list()
    for j in range(beam_width):
        most_likely_cap.append(list())
        most_likely_cap[j] = [[caption_train_tokenizer.index_word[most_likely_seq[j]]]]

    for i in range(max_length):
        temp_prob = np.zeros((beam_width, vocab_size))
        for j in range(beam_width):
            if most_likely_cap[j][-1] != ['end']:  # if not terminated
                num_words = len(most_likely_cap[j])
                sequence = caption_train_tokenizer.texts_to_sequences(most_likely_cap[j])
                sequence = pad_sequences(np.transpose(sequence), maxlen=max_length)
                model_softMax_output = pred_model.predict([photo,sequence], verbose=0)
                # update most likily prob
                temp_prob[j,] = (1 / num_words) * (most_likely_prob[j]*(num_words-1) + np.log(model_softMax_output))
            else:
                temp_prob[j,] = most_likely_prob[j] + np.zeros(vocab_size) - np.inf
                temp_prob[j,0] = most_likely_prob[j]

        x_idx, y_idx = np.unravel_index(temp_prob.flatten().argsort()[-beam_width:], temp_prob.shape)

        most_likely_cap_temp = list()
        for j in range(beam_width):
            most_likely_prob[j] = temp_prob[x_idx[j],y_idx[j]]
            most_likely_cap_temp.append(list())
            most_likely_cap_temp[j] = most_likely_cap[x_idx[j]].copy()
            if most_likely_cap_temp[j][-1] != ['end']:
                most_likely_cap_temp[j].append([caption_train_tokenizer.index_word[y_idx[j]]])

        most_likely_cap = most_likely_cap_temp.copy()

        finished = True
        for j in range(beam_width):
            if most_likely_cap_temp[j][-1] != ['end']:
                finished = False

        if finished:
            break

    final_caption = list()

    for j in range(beam_width):
        final_caption.append(' '.join(flatten(most_likely_cap[j][0:-1])))

    return final_caption, most_likely_prob


"""
CODE FROM NOTEBOOK
</END>
"""


def main():
    # Constants.
    caption_train_tokenizer_path = 'pretrained/caption_train_tokenizer.pkl'
    model_path = 'pretrained/modelConcat_1_5.h5'
    dataset_prefix = 'data/MS-COCO/train/'
    use_beam_search = False
    NUM_IMAGES = 100

    # Pick NUM_IMAGES random images from the MS-COCO dataset.
    # Because our model was trained on the Flickr-8k dataset.
    print("Loading {} random MS-COCO 2014 train dataset images...".format(NUM_IMAGES))
    images_filepaths = []
    with open('data/MS-COCO/train/captions.json', 'rb') as f:
        mscoco_info = json.load(f)
        image_infos = list(mscoco_info['images'])
        random.shuffle(image_infos)
        images_filepaths = [o["file_name"] for o in image_infos[0:NUM_IMAGES]]
        image_ids = [o["id"] for o in image_infos[0:NUM_IMAGES]]

        queries_captions = []
        expected_retrieved_image_id = []
        for o in mscoco_info['annotations']:
            if o['image_id'] in image_ids:
                queries_captions.append(o['caption'])
                expected_retrieved_image_id.append(o['image_id'])

    # Load the model.
    print("Loading model...")
    base_model = VGG16(include_top=True)
    feature_extract_pred_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    print("Loading tokenizer...")
    caption_train_tokenizer = load(open(caption_train_tokenizer_path, 'rb'))
    max_length = 33
    pred_model = load_model(model_path)

    # Get the captions for the 100 images.
    # get the path/directory
    vocab_size = 7506
    beam_width = 10
    max_length = 33

    image_captions_beam = dict()

    print("Generating captions...")
    for image_fp in tqdm.tqdm(images_filepaths):
        photo = extract_feature(feature_extract_pred_model, dataset_prefix + 'images/' + image_fp)
        if use_beam_search:
            beam_captions, prob = generate_caption_beam(pred_model, caption_train_tokenizer, photo, max_length, vocab_size, beam_width)
            generated_caption = ''.join(beam_captions[np.argmax(prob)])
        else:
            generated_caption = generate_caption(pred_model, caption_train_tokenizer, photo, max_length)
        image_captions_beam[image_fp] = generated_caption

    # For each image we have 5 ground truth captions.
    # If the system is well-designed, using a ground truth caption should return the associated image.
    # Use each caption of the 100 images (so 500 captions in total) to compute the scores.
    print("Computing scores...")
    scores = np.zeros((len(queries_captions), len(image_captions_beam)))

    for i, query_caption in enumerate(tqdm.tqdm(queries_captions)):
        for j, image_fp in enumerate(images_filepaths):
            scores[i, j] = metrics.compute_bleu(image_captions_beam[image_fp], tokenizer_fn(query_caption))

    # From the scores for all the 500 queries, compute the retrieve@k metric.
    # Retrieve@k metric = is the ground-truth image within the top-k scores?
    k = range(1, 20)

    index_of_correct_image = [image_ids.index(o) for o in expected_retrieved_image_id]
    retrieve_at_k = np.zeros((len(k), 1))
    for i, k_i in enumerate(k):
        retrieve_at_k[i] = np.mean(
            np.any(
                scores[:, index_of_correct_image] > np.sort(scores, axis=1)[:, -k_i],
                axis=1
            )
        )

    """
    retrieve_at_k = np.zeros((len(k), 1))
    for k_ in k:

        # Retrieve the top-k indices (in the image_captions_beam array) for each caption.
        topk_retrieved_indices = np.argpartition(scores, -k)[-k:]

        # Check if the expected_retrieved_image_id is in the top-k indices.
        for i, ith_topk_retrieved in enumerate(topk_retrieved_indices):
            # expected_retrieved_image_id[i] is the image id of the ith query.
            # ith_topk_retrieved is the index of the ith query in the image_captions_beam array.
            image_ids_ith_topk_retrieved = [image_ids[x] for x in ith_topk_retrieved]

            if expected_retrieved_image_id[i] in image_ids_ith_topk_retrieved:
                retrieve_at_k[k_] += 1

        retrieve_at_k[k_] /= len(queries_captions)
    """

    print("Retrieval@k:", retrieve_at_k)

    plt.plot(k, retrieve_at_k)
    plt.xlabel('k')
    plt.ylabel('Retrieval@k')
    plt.show()

if __name__ == "__main__":
    main()
