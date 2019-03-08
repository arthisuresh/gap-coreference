import pandas as pd
import spacy
import pprint as pprint
import numpy as np
import torch as torch
import os
import random
import math

PROCESSED_DATA_DIR = "processed_data"

def bin_vector(value):
    assert(value >= 0)
    if value <= 4:
        idx = value
    else:
        idx = min(math.floor(math.log(value, 2)), 6) + 3
    vec = np.zeros(10)
    vec[idx] = 1
    return vec

def generate_embedding_vector(passage_annotated, mention_start, mention_end):
    noun_chunk = [chunk for chunk in passage_annotated.noun_chunks if (mention_start in range(chunk.start_char, chunk.end_char+1) or mention_end in range(chunk.start_char, chunk.end_char+1))]
    words = [word for word in passage_annotated if word.idx >= mention_start and word.idx <= mention_end]
    dependency_parent = random.choice([word.head.vector for word in words])
    head_word = random.choice([word.vector for word in words])
    if noun_chunk:
        noun_chunk = noun_chunk[0]
        dependency_parent = noun_chunk.root.head.vector
        head_word = noun_chunk.root.vector

    sentence = [sent for sent in passage_annotated.sents if (mention_start in range(sent.start_char, sent.end_char+1) or mention_end in range(sent.start_char, sent.end_char+1))]

    first_word = words[0].vector
    last_word = words[-1].vector
    two_preceding_words_list = [w.vector for w in passage_annotated if w.i in range(words[0].i-2, words[0].i)]
    two_preceding_words_list = two_preceding_words_list + [np.zeros(300)]*(2-len(two_preceding_words_list))
    two_preceding_words = np.hstack(two_preceding_words_list)
    two_following_words_list = [w.vector for w in passage_annotated if w.i in range(words[-1].i+1, words[-1].i+3)]
    two_following_words_list = two_following_words_list + [np.zeros(300)]*(2-len(two_following_words_list))
    two_following_words = np.hstack(two_following_words_list)
    avg_word_embedding_prev_5 = np.mean(np.vstack([w.vector for w in passage_annotated if w.i in range(words[0].i-5, words[0].i)] or [np.zeros(300)]*5), axis=0)
    avg_word_embedding_next_5 = np.mean(np.vstack([w.vector for w in passage_annotated if w.i in range(words[-1].i+1, words[-1].i+6)] or [np.zeros(300)]*5), axis=0)
    avg_word_embedding = np.mean(np.vstack([w.vector for w in passage_annotated if w.i in range(words[0].i, words[-1].i+1)]), axis=0)
    avg_sentence_embedding = np.mean(np.vstack([w.vector for w in sentence]), axis=0)
    embedding_vector = np.hstack((head_word, dependency_parent, first_word, last_word, two_preceding_words, two_following_words, avg_word_embedding_prev_5, avg_word_embedding_next_5, avg_word_embedding, avg_sentence_embedding))

    return embedding_vector

def generate_binary_and_distance_features(passage_annotated, mention_start, mention_end, pronoun_start, pronoun_end):
    words = [word for word in passage_annotated if word.idx >= mention_start and word.idx <= mention_end]
    pronoun_words = [word for word in passage_annotated if word.idx >= pronoun_start and word.idx <= pronoun_end]
    # distance and length features
    n_words = len(words)
    n_words_bucketed = bin_vector(n_words)
    n_chars = mention_end - mention_start
    n_chars_bucketed = bin_vector(n_chars)
    n_words_to_pronoun = abs(pronoun_words[0].i - words[0].i)
    n_words_to_pronoun_bucketed = bin_vector(n_words_to_pronoun)
    n_chars_to_pronoun = abs(pronoun_start - mention_start)
    n_chars_to_pronoun_bucketed = bin_vector(n_chars_to_pronoun)

    # binary features
    noun_chunk = [(i, chunk) for i, chunk in enumerate(passage_annotated.noun_chunks) if (mention_start in range(chunk.start_char, chunk.end_char+1) or mention_end in range(chunk.start_char, chunk.end_char+1))]
    mention_pos = 1
    contained_within = 0
    if noun_chunk:
        noun_chunk = noun_chunk[0]
        mention_pos = noun_chunk[0]/len([chunk for chunk in passage_annotated.noun_chunks])
        if noun_chunk[1].text != passage_annotated.text[mention_start:mention_end]:
            contained_within = 1

    return np.hstack((np.array([mention_pos, contained_within, n_words, n_chars, n_words_to_pronoun, n_chars_to_pronoun]), \
        n_words_bucketed, n_chars_bucketed, n_words_to_pronoun_bucketed, n_chars_to_pronoun_bucketed))

def get_padded_word_embeddings_and_labels(dataset):
    filename='gap-{}.tsv'.format(dataset)
    data = pd.read_csv(filename, delimiter='\t')
    num_samples=len(data)
    print("{num_samples} samples read from {filename}".format(num_samples=num_samples, filename=filename))
    nl
    emb_matrix_list = []
    output_vector_list = []
    for i, row in data.iterrows():
        passage = row['Text']
        pronoun = row['Pronoun']
        pronoun_start = row['Pronoun-offset']

        passage_annotated = nlp(passage)
        embs = [w.vector for w in passage_annotated]

        # embeddings for passage
        mention_A = row['A']
    
    pad_sents()


def get_feature_embeddings_and_labels(dataset):
    filename='gap-{}.tsv'.format(dataset)
    data = pd.read_csv(filename, delimiter='\t')
    num_samples=len(data)
    print("{num_samples} samples read from {filename}".format(num_samples=num_samples, filename=filename))
    nlp = spacy.load('en_core_web_lg')
    emb_matrix_list = []
    output_vector_list = []
    for i, row in data.iterrows():
        sent = row['Text']
        pronoun = row['Pronoun']
        pronoun_start = row['Pronoun-offset']
        pronoun_end = pronoun_start + len(pronoun)

        mention_A = row['A']
        mention_A_start = row['A-offset'] 
        mention_A_end = mention_A_start + len(mention_A)

        mention_B = row['B']
        mention_B_start = row['B-offset'] 
        mention_B_end = mention_B_start + len(mention_B)

        A_coref = row['A-coref']
        B_coref = row['B-coref']
        output_vector = []
        if A_coref is True:
            output_vector = 0
        elif B_coref is True:
            output_vector = 1
        else:
            output_vector = 2

        passage_annotated = nlp(sent)
        
        # segment the sentence in terms of noun chunks and words in the mention
        embedding_vector_A = generate_embedding_vector(passage_annotated, mention_A_start, mention_A_end)
        embedding_vector_B = generate_embedding_vector(passage_annotated, mention_B_start, mention_B_end)
        embedding_vector_pronoun = generate_embedding_vector(passage_annotated, pronoun_start, pronoun_end)
        position_features_A = generate_binary_and_distance_features(passage_annotated, mention_A_start, mention_A_end, pronoun_start, pronoun_end)
        position_features_B = generate_binary_and_distance_features(passage_annotated, mention_B_start, mention_B_end, pronoun_start, pronoun_end)
        document_embedding = np.mean(np.vstack([word.vector for word in passage_annotated]), axis=0)

        feature_vector = np.hstack((embedding_vector_A, embedding_vector_B, embedding_vector_pronoun, position_features_A, position_features_B, document_embedding))
        emb_matrix_list.append(feature_vector)
        output_vector_list.append(output_vector)

    emb_matrix = np.vstack(emb_matrix_list)
    output_matrix = np.array(output_vector_list)
    return (torch.from_numpy(emb_matrix), torch.from_numpy(output_matrix))
   
def prepare_data(dataset, reload=False):
    # prepare training features
    X_data_output_path = os.path.join(
        PROCESSED_DATA_DIR,
        'processed_{dataset}_X.pt'.format(dataset=dataset)
    )
    Y_data_output_path = os.path.join(
        PROCESSED_DATA_DIR,
        'processed_{dataset}_Y.pt'.format(dataset=dataset)
    )
    if not os.path.exists(X_data_output_path) \
        or not os.path.exists(Y_data_output_path) or reload:
        X_data, Y_data = get_feature_embeddings_and_labels(dataset)
        print(X_data.size())
        print(Y_data.size())
        torch.save(X_data, X_data_output_path)
        torch.save(Y_data, Y_data_output_path)

def load_data(dataset):
    X_data_path = os.path.join(
        PROCESSED_DATA_DIR,
        'processed_{dataset}_X.pt'.format(dataset=dataset)
    )
    Y_data_path = os.path.join(
        PROCESSED_DATA_DIR,
        'processed_{dataset}_Y.pt'.format(dataset=dataset)
    )
    filename='gap-{}.tsv'.format(dataset)
    data = pd.read_csv(filename, delimiter='\t')
    ids = data['ID'].tolist()
    print(ids)
    return ids, torch.load(X_data_path).float(), torch.load(Y_data_path).long()