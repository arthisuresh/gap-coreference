import pandas as pd
import spacy
import pprint as pprint
import numpy as np
import torch as torch
import os
import random
import math
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

PROCESSED_DATA_DIR = "processed_data"
MODEL_NAME = "BERT_Attention"
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
nlp = spacy.load('en_core_web_lg')

def bin_vector(value):
    assert(value >= 0)
    if value <= 4:
        idx = value
    else:
        idx = min(math.floor(math.log(value, 2)), 6) + 3
    vec = np.zeros(10)
    vec[idx] = 1
    return list(vec)


def get_bert_embeddings(passage_text):
    passage_text_processed = nlp(passage_text)
    passage_with_separators = ' '.join(['[CLS]'] + [sent.text + ' [SEP]' for sent in passage_text_processed.sents])
    passage_with_separators_tokenized = tokenizer.tokenize(passage_with_separators)
    
    model.eval()
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(passage_with_separators_tokenized)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor)

    # remove separator tokens
    indices = [i for i, w in enumerate(passage_with_separators_tokenized) if (w not in ['[CLS]', '[SEP]'])]
    nonseparator_tokens = [w for i, w in enumerate(passage_with_separators_tokenized) if (w not in ['[CLS]', '[SEP]'])]
    nonseparators = torch.squeeze(encoded_layers[-1])[indices][:]

    attn_vectors_per_word = []
    encountered_words = []
    i = 0
    # print([word.text for word in passage_text_processed])
    carry_over = None
    had_carry_over = False
    # print([word for word in passage_text_processed])
    for w_i, word in enumerate(passage_text_processed):
        # print("WORD is {word}".format(word=word))
        word = word.text.lower()
        first_attention_vector = nonseparators[i]
        current_word = ''
        if word == ' ':
            attn_vectors_per_word.append(first_attention_vector)
            continue
        if carry_over:
            current_word = carry_over
            carry_over = None
        while current_word[:len(word)] != word:
            current_token = nonseparator_tokens[i]
            current_word += (current_token if (current_token[:2] != '##') else current_token[2:])
            # print(current_word[:len(word)])
            i += 1
        encountered_words.append(current_word)
        if not had_carry_over:
            attn_vectors_per_word.append(first_attention_vector)
        else:
            had_carry_over = False
        if len(current_word) > len(word):
            attn_vectors_per_word.append(first_attention_vector)
            carry_over = current_word[len(word):]
            had_carry_over = True
    output = torch.stack(attn_vectors_per_word)
    assert len([word for word in passage_text_processed]) == output.size()[0]
    return output


def generate_bert_vector(passage_text, passage_annotated, mention_start, mention_end):
    noun_chunk = [chunk for chunk in passage_annotated.noun_chunks if (mention_start in range(chunk.start_char, chunk.end_char+1) or mention_end in range(chunk.start_char, chunk.end_char+1))]
    words_in_mention = [word for i, word in enumerate(passage_annotated) if word.idx >= mention_start and word.idx <= mention_end]
    word_indices_of_mention = [i for i, word in enumerate(passage_annotated) if word.idx >= mention_start and word.idx <= mention_end]
    word_embeddings = get_bert_embeddings(passage_text)
    mention_embeddings = word_embeddings[word_indices_of_mention, :]
    dependency_parent = random.choice([word_embeddings[word.head.i, :] for word in words_in_mention])
    head_word = random.choice([word_embeddings[word.i, :] for word in words_in_mention])
    # print(head_word.size())
    if noun_chunk:
        noun_chunk = noun_chunk[0]
        dependency_parent = word_embeddings[noun_chunk.root.head.i, :]
        head_word = word_embeddings[noun_chunk.root.i,:]
    # print(head_word.size())
    sentence = [sent for sent in passage_annotated.sents if (mention_start in range(sent.start_char, sent.end_char+1) or mention_end in range(sent.start_char, sent.end_char+1))]

    first_word = mention_embeddings[0, :]
    last_word = mention_embeddings[-1, :]

    two_preceding_words_skeleton = torch.zeros(2, 768)
    two_preceding_words = word_embeddings[max(word_indices_of_mention[0]-2, 0):word_indices_of_mention[0], :]
    two_preceding_words_skeleton[(2-two_preceding_words.size()[0]):2,:] = two_preceding_words
    two_preceding_words_skeleton = torch.reshape(two_preceding_words_skeleton, (-1,))
    # print(two_preceding_words_skeleton.size())

    two_following_words_skeleton = torch.zeros(2, 768)
    two_following_words = word_embeddings[word_indices_of_mention[-1]:word_indices_of_mention[-1]+2, :]
    two_following_words_skeleton[0:two_following_words.size()[0], :] = two_following_words
    two_following_words_skeleton = torch.reshape(two_following_words_skeleton, (-1,))
    # print(two_following_words_skeleton.size())

    avg_word_embedding_prev_5 = torch.mean(word_embeddings[max(word_indices_of_mention[0]-5, 0):word_indices_of_mention[0], :], dim=0)
    # print(avg_word_embedding_prev_5.size())
    if tuple(avg_word_embedding_prev_5.size()) != (768,):
        avg_word_embedding_prev_5 = torch.zeros(768)
    avg_word_embedding_next_5 = torch.mean(word_embeddings[word_indices_of_mention[-1]:word_indices_of_mention[-1]+5, :], dim=0)
    if tuple(avg_word_embedding_prev_5.size()) != (768,):
        avg_word_embedding_next_5 = torch.zeros(768)
    avg_word_embedding = torch.mean(mention_embeddings, dim=0)
    # print(avg_word_embedding_prev_5.size())
    # print(avg_word_embedding_next_5.size())
    # print(avg_word_embedding.size())

    avg_sentence_embedding = torch.mean(word_embeddings, dim=0)
    # print(avg_sentence_embedding.size())
    # print("SIZE")
    # for t in (torch.unsqueeze(head_word, 0), torch.unsqueeze(dependency_parent, 0), torch.unsqueeze(first_word, 0), torch.unsqueeze(last_word, 0), torch.unsqueeze(two_preceding_words_skeleton, 0), torch.unsqueeze(two_following_words_skeleton, 0), torch.unsqueeze(avg_word_embedding_prev_5, 0), torch.unsqueeze(avg_word_embedding_next_5, 0), torch.unsqueeze(avg_word_embedding, 0), torch.unsqueeze(avg_sentence_embedding, 0)):
    #     print(t.size())
    embedding_vector = torch.cat((torch.unsqueeze(head_word, 0), torch.unsqueeze(dependency_parent, 0), torch.unsqueeze(first_word, 0), torch.unsqueeze(last_word, 0), torch.unsqueeze(two_preceding_words_skeleton[:768], 0), torch.unsqueeze(two_preceding_words_skeleton[768:], 0), torch.unsqueeze(two_following_words_skeleton[:768], 0), torch.unsqueeze(two_following_words_skeleton[768:], 0), torch.unsqueeze(avg_word_embedding_prev_5, 0), torch.unsqueeze(avg_word_embedding_next_5, 0), torch.unsqueeze(avg_word_embedding, 0), torch.unsqueeze(avg_sentence_embedding, 0)), dim=1)
    embedding_vector = torch.cat((head_word, dependency_parent, first_word, last_word, two_preceding_words_skeleton[:768], two_preceding_words_skeleton[768:], two_following_words_skeleton[:768], two_following_words_skeleton[768:], avg_word_embedding_prev_5, avg_word_embedding_next_5, avg_word_embedding, avg_sentence_embedding), dim=0)
    # print(embedding_vector.size())
    return embedding_vector

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
    # print([mention_pos, contained_within, n_words, n_chars, n_words_to_pronoun, n_chars_to_pronoun] + n_words_bucketed + n_chars_bucketed + n_words_to_pronoun_bucketed + n_chars_to_pronoun_bucketed)
    return torch.Tensor([mention_pos, contained_within, n_words, n_chars, n_words_to_pronoun, n_chars_to_pronoun] + n_words_bucketed + n_chars_bucketed + n_words_to_pronoun_bucketed + n_chars_to_pronoun_bucketed)

def get_padded_word_embeddings_and_labels(dataset):
    filename = 'gap-{}.tsv'.format(dataset)
    data = pd.read_csv(filename, delimiter='\t')
    num_samples = len(data)
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


def get_feature_embeddings_and_labels(dataset, setting='pairs'):
    filename='gap-{}.tsv'.format(dataset)
    data = pd.read_csv(filename, delimiter='\t')
    num_samples=len(data)
    print("{num_samples} samples read from {filename}".format(num_samples=num_samples, filename=filename))
    emb_matrix_list = []
    output_vector_list = []
    mention_names_list = []
    true_mentions = []
    for i, row in data.iterrows():
        print('Loading {}...'.format(row['ID']))
        # Parse each row and extract the candidate mentions, and the ambiguous pronoun
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

        passage_annotated = nlp(sent)
        feature_vectors = []
        output_vector = []
        # there are two possible settings:
        # 1. having knowledge of the two mentions A and B in the input to our model ('classification', since this
        #       essentially a 3-class classification task)
        # 2. not having knowledge of the two mentions A and B until evaluation ('pairs')
        if setting == 'classification':
            if A_coref is True:
                output_vectors = 0
            elif B_coref is True:
                output_vector = 1
            else:
                output_vector = 2
            # generate features for mention A, B, the pronoun, position features, the overall document embedding,
            # and feature vectors
            embedding_vector_A = generate_bert_vector(sent, passage_annotated, mention_A_start, mention_A_end)
            embedding_vector_B = generate_bert_vector(sent, passage_annotated, mention_B_start, mention_B_end)
            embedding_vector_pronoun = generate_bert_vector(sent, passage_annotated, pronoun_start, pronoun_end)
            position_features_A = generate_binary_and_distance_features(passage_annotated, mention_A_start, mention_A_end, pronoun_start, pronoun_end)
            position_features_B = generate_binary_and_distance_features(passage_annotated, mention_B_start, mention_B_end, pronoun_start, pronoun_end)
            # document_embedding = np.mean(np.vstack([word.vector for word in passage_annotated]), axis=0)
            feature_vectors = torch.cat((embedding_vector_A, embedding_vector_B, embedding_vector_pronoun, position_features_A, position_features_B))
        elif setting == 'pairs':
            feature_vectors = []
            output_vectors = []
            mention_names = []
            emb_vector_pronoun = generate_bert_vector(sent, passage_annotated, pronoun_start, pronoun_end)
            # document_embedding = np.mean(np.vstack([word.vector for word in passage_annotated]), axis=0)
            # print(sent)
            # print((pronoun, mention_A, A_coref, mention_B, B_coref))
            # print([ent for ent in passage_annotated.ents])
            for ent in passage_annotated.ents:
                emb_vector_candidate = generate_bert_vector(sent, passage_annotated, ent.start_char, ent.end_char)
                position_features = generate_binary_and_distance_features(passage_annotated, ent.start_char, ent.end_char, pronoun_start, pronoun_end)
                # print(position_features.size())
                feature_vector = torch.cat((emb_vector_candidate, emb_vector_pronoun, position_features))
                feature_vectors.append(feature_vector) 
                output_vectors.append(1 if ((ent.text == mention_A and A_coref) or (ent.text == mention_B and B_coref)) else 0)
                mention_names.append(ent.text)
            if not (A_coref or B_coref):
                output_vectors.append(1)
            else:
                output_vectors.append(0)
        #print(feature_vectors)
        truth_label = 'Neither'
        if A_coref:
            truth_label = 'A'
        elif B_coref:
            truth_label = 'B'
        true_mentions.append([mention_A, mention_B, truth_label])
        emb_matrix_list.append(torch.stack(feature_vectors))
        output_vector_list.append(torch.Tensor(output_vectors))
        mention_names_list.append(mention_names)
    return (emb_matrix_list, output_vector_list, mention_names_list, true_mentions)
   

def prepare_data(dataset, setting='pairs', reload=False):
    # prepare training features
    X_data_output_path = os.path.join(
        PROCESSED_DATA_DIR,
        '{setting}_bert_{dataset}_X.pt'.format(dataset=dataset, setting=setting)
    )
    Y_data_output_path = os.path.join(
        PROCESSED_DATA_DIR,
        '{setting}_bert_{dataset}_Y.pt'.format(dataset=dataset, setting=setting)
    )
    E_data_output_path = os.path.join(
        PROCESSED_DATA_DIR,
        '{setting}_bert_{dataset}_entities.pt'.format(dataset=dataset, setting=setting)
    )
    M_data_output_path = os.path.join(
        PROCESSED_DATA_DIR,
        '{setting}_bert_{dataset}_gold_mentions.pt'.format(dataset=dataset, setting=setting)
    )
    if not os.path.exists(X_data_output_path) \
        or not os.path.exists(Y_data_output_path) or not os.path.exists(M_data_output_path) or reload:
        X_data, Y_data, ents, mentions = get_feature_embeddings_and_labels(dataset, setting)
        torch.save(X_data, X_data_output_path)
        torch.save(Y_data, Y_data_output_path)
        torch.save(ents, E_data_output_path)
        torch.save(mentions, M_data_output_path)


def load_data(dataset, setting='pairs'):
    X_data_path = os.path.join(
        PROCESSED_DATA_DIR,
        '{setting}_bert_{dataset}_X.pt'.format(dataset=dataset, setting=setting)
    )
    Y_data_path = os.path.join(
        PROCESSED_DATA_DIR,
        '{setting}_bert_{dataset}_Y.pt'.format(dataset=dataset, setting=setting)
    )
    E_data_path = os.path.join(
        PROCESSED_DATA_DIR,
        '{setting}_bert_{dataset}_entities.pt'.format(dataset=dataset, setting=setting)
    )
    M_data_path = os.path.join(
        PROCESSED_DATA_DIR,
        '{setting}_bert_{dataset}_gold_mentions.pt'.format(dataset=dataset, setting=setting)
    )
    filename='gap-{}.tsv'.format(dataset)
    data = pd.read_csv(filename, delimiter='\t')
    ids = data['ID'].tolist()
    return ids, torch.load(X_data_path), torch.load(Y_data_path), torch.load(E_data_path), torch.load(M_data_path)
sent = "John Nixon (May 10, 1815 - June 3, 1899), English mining engineer and colliery proprietor, was born at Barlow, Durham, the son of a farmer. He was educated at the village school, and at an academy in Newcastle-on-Tyne, where he distinguished himself in mathematics."
sent = "Charlotte Church  - Singer and TV chat show host moved to the village in July 2010 Hannah Mills - British sailor, Olympic silver medalist at London 2012 Olympic Games in the Women's 470 Class she went on to win the gold medal at the 2016 Olympic Games in Rio de Janeiro, from Dinas Powys"
# sent_p = nlp(sent)
# # get_bert_embeddings("By this time, Karen Blixen had separated from her husband, and after their divorce in 1925, Finch Hatton moved into her house and began leading safaris for wealthy sportsmen. Among his clients were Marshall Field Jr and Edward, Prince of Wales. According to the author Mary Lovell, in 1930 Finch Hatton began a love affair with Beryl Markham, who was working as a race-horse trainer in Nairobi and the surrounding area. Later, she would become known as a pioneer flyer herself (Markham attributed her interest in flying to her association with Tom Campbell Black).")
# generate_bert_vector(sent, sent_p, 83, 94)
