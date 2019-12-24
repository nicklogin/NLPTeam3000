import pandas as pd
import xml.etree.ElementTree as et
import os
import re
import urllib.request
import keras.backend
import keras.models
import matplotlib.pyplot as plt
from matplotlib import cm, transforms
import numpy as np
import pickle
import time

xtree = et.parse(urllib.request.urlopen("https://raw.githubusercontent.com/nicklogin/NLPTeam3000/master/development/SentiRuEval_rest_train.xml"))
root = xtree.getroot()
data = []
columns = ['id', 'food', 'service', 'text']
for review in root:
    text_id = int(review.attrib['id'])
    
    scores = review.find('scores')
    
    food = int(scores.find('food').text)
    service = int(scores.find('service').text)
    
    text = review.find('text').text

    new_food = 0
    new_service = 0

    if food > 5:
        new_food = 1

    if service > 5:
        new_service = 1
    
    data.append({'id': text_id,
                'food': new_food,
                'service': new_service,
                'text': text})
    

df = pd.DataFrame(data, columns=columns)
df = df.set_index('id')
print(root[0].find('text').text)

test_txts = []
food_negs = []
food_poss = []
service_negs = []
service_poss = []



for file in os.listdir("Downloads/NLPTeam3000-master-y/conllu_data"):
    if file.endswith('.tsv'):
        file_text = open(os.path.join("Downloads/NLPTeam3000-master-y/conllu_data", file), 'r', encoding='utf-8').read()
        text = ''
        sentences = []
        for l in file_text.splitlines():
            if '# text = ' in l:
                sentences.append(l.replace('# text = ', ''))
                text += l.replace('# text = ', '')
    test_txts.append(text)


    food_neg = []
    food_pos = []
    service_neg = []
    service_pos = []


    markup = open(os.path.join('Downloads/NLPTeam3000-master-y/разметка_финал', file[:-4]+".tsv"), 'r', encoding='utf-8').read()
    for line in markup.splitlines():
        line = line.strip()
        if line:
            l = line.split('\t')[0]
            ws = []
            for n in line.split('\t')[1].split(','):
                ws.append(sentences[int(l) - 1][int(n) - 1])

            if line.split('\t')[2] == 'Food' and int(line.split('\t')[3]) == 0:
                food_neg.append(ws)
            elif line.split('\t')[2] == 'Food' and int(line.split('\t')[3]) == 1:
                food_pos.append(ws)
            elif line.split('\t')[2] == 'Service' and int(line.split('\t')[3]) == 0:
                service_neg.append(ws)
            elif line.split('\t')[2] == 'Service' and int(line.split('\t')[3]) == 1:
                service_pos.append(ws)

    food_negs.append(food_neg)
    food_poss.append(food_pos)
    service_negs.append(service_neg)
    service_poss.append(service_pos)

print(len(test_txts))
print(len(food_negs))
print(len(food_poss))
print(len(service_negs))
print(len(service_poss))

texts = []
foods = []
services = []

for review in root:
    
    scores = review.find('scores')
    
    food = int(scores.find('food').text)
    service = int(scores.find('service').text)
    
    texts.append(review.find('text').text)

    new_food = 0
    new_service = 0

    if food > 5:
        new_food = 1

    if service > 5:
        new_service = 1
    
    foods.append(new_food)
    services.append(new_service)

print(len(texts))
print(len(foods))
print(len(services))

def max_length(texts):
    return max(len(t) for t in texts)

from gensim.models import KeyedVectors
w2v_modelf = KeyedVectors.load_word2vec_format('model.bin', binary=True)
w2v_models = KeyedVectors.load_word2vec_format('model.bin', binary=True)

from sklearn.model_selection import train_test_split
scores_trainf, scores_valf, texts_trainf, texts_valf = train_test_split(
    foods[:18943], texts[:18943], test_size=0.3)

from sklearn.model_selection import train_test_split
scores_trains, scores_vals, texts_trains, texts_vals = train_test_split(
    services[:18943], texts[:18943], test_size=0.3)

from collections import Counter
from itertools import chain

MAX_LENf = max(max_length(texts_trainf), max_length(texts_valf))
MAX_LENs = max(max_length(texts_trains), max_length(texts_vals))

def load_datasetf(lines, embedding_dim, num_examples=None):
    prep = lines[:num_examples]
    vocab = Counter()
    x_tensor = np.zeros((len(prep), MAX_LENf, embedding_dim))
    for i, text in enumerate(prep):
        for j, w in enumerate(text):
            try:
                x_tensor[i, j, :] = w2v_modelf[w]
            except KeyError:
                pass
        vocab[w] += 1
    return x_tensor, vocab

def load_datasets(lines, embedding_dim, num_examples=None):
    prep = lines[:num_examples]
    vocab = Counter()
    x_tensor = np.zeros((len(prep), MAX_LENs, embedding_dim))
    for i, text in enumerate(prep):
        for j, w in enumerate(text):
            try:
                x_tensor[i, j, :] = w2v_models[w]
            except KeyError:
                pass
        vocab[w] += 1
    return x_tensor, vocab

input_tensor_trainf, inp_vocab_trainf = load_datasetf(texts_trainf, w2v_modelf.vector_size)
input_tensor_valf, inp_vocab_valf = load_datasetf(texts_valf, w2v_modelf.vector_size)
input_tensor_trains, inp_vocab_trains = load_datasets(texts_trains, w2v_models.vector_size)
input_tensor_vals, inp_vocab_vals = load_datasets(texts_vals, w2v_models.vector_size)

print(w2v_modelf.vector_size)
print(w2v_models.vector_size)

print(input_tensor_trainf.shape)
print(input_tensor_trains.shape)

embedding_dimf = w2v_modelf.vector_size
inp_vocabf = inp_vocab_trainf + inp_vocab_valf
vocab_inp_sizef = len(inp_vocabf)+1

embedding_dims = w2v_models.vector_size
inp_vocabs = inp_vocab_trains + inp_vocab_vals
vocab_inp_sizes = len(inp_vocabs)+1

from innvestigate.utils.tests.networks import base as network_base
def build_network(max_len, voc_size, embedding_dim, output_n, activation=None, dense_unit=256, dropout_rate=0.25):
    if activation:
        activation = "relu"

    net = {}
    net["in"] = keras.Input(shape=[1, max_len, embedding_dim])
    net["conv"] = keras.layers.Conv2D(filters=100, kernel_size=(1,2), strides=(1, 1), padding='valid')(net["in"])
    net["pool"] = keras.layers.MaxPooling2D(pool_size=(1, max_len-1), strides=(1,1))(net["conv"])
    net["out"] = network_base.dense_layer(keras.layers.Flatten()(net["pool"]), units=output_n, activation=activation)
    net["sm_out"] = network_base.softmax(net["out"])


    net.update({
        "input_shape": [1, max_len, embedding_dim],
        "output_n": output_n,
    })
    return net

netf = build_network(MAX_LENf, vocab_inp_sizef, embedding_dimf, 2)
model_without_softmaxf = keras.models.Model(inputs=netf['in'], outputs=netf['out'])
model_with_softmaxf = keras.models.Model(inputs=netf['in'], outputs=netf['sm_out'])

nets = build_network(MAX_LENs, vocab_inp_sizes, embedding_dims, 2)
model_without_softmaxs = keras.models.Model(inputs=nets['in'], outputs=nets['out'])
model_with_softmaxs = keras.models.Model(inputs=nets['in'], outputs=nets['sm_out'])

print(model_without_softmaxs.summary())
print(model_without_softmaxf.summary())

def to_one_hot(y):
    return keras.utils.to_categorical(y, 2)

def train_modelf(model, epochs=20):
    
    x_train = np.expand_dims(input_tensor_trainf, axis=1)
    y_train = to_one_hot(scores_trainf)
    
    x_val = np.expand_dims(input_tensor_valf, axis=1)
    y_val = to_one_hot(scores_valf)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=50,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_val, y_val),
                        shuffle=True
                       )

def train_models(model, epochs=20):
    
    x_train = np.expand_dims(input_tensor_trains, axis=1)
    y_train = to_one_hot(scores_trains)
    
    x_val = np.expand_dims(input_tensor_vals, axis=1)
    y_val = to_one_hot(scores_vals)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=50,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_val, y_val),
                        shuffle=True
                       )

train_modelf(model_with_softmaxf, epochs=10)
train_models(model_with_softmaxs, epochs=10)

model_without_softmaxf.set_weights(model_with_softmaxf.get_weights())
model_without_softmaxs.set_weights(model_with_softmaxs.get_weights())

methods = ['gradient', 'lrp.z', 'lrp.alpha_2_beta_1', 'pattern.attribution']
kwargs = [{}, {}, {}, {'pattern_type': 'relu'}]

import investigate
analyzersf = []

for method, kws in zip(methods, kwargs):
    analyzerf = innvestigate.create_analyzer(method, model_without_softmaxf, **kws)
    analyzerf.fit(np.expand_dims(input_tensor_trainf, axis=1), batch_size=256, verbose=1)
    analyzersf.append(analyzerf)
    
analyzerss = []

for method, kws in zip(methods, kwargs):
    analyzers = innvestigate.create_analyzer(method, model_without_softmaxs, **kws)
    analyzers.fit(np.expand_dims(input_tensor_trains, axis=1), batch_size=256, verbose=1)
    analyzerss.append(analyzers)

def analyze_scoresf(X, Y, ridx):
    max_len = max_length(input_tensor_trainf)

    analysis = np.zeros([len(analyzers), 1, max_len])
    x, y = X[ridx], Y[ridx]
    t_start = time.time()
    x = x.reshape((1, 1, max_len, embedding_dimf))
    presm = model_without_softmaxf.predict_on_batch(x)[0] #forward pass without softmax
    prob = model_with_softmaxf.predict_on_batch(x)[0] #forward pass with softmax
    y_hat = prob.argmax()
  
    for aidx, analyzer in enumerate(analyzersf):
        a = np.squeeze(analyzer.analyze(x))
        a = np.sum(a, axis=1)
        analysis[aidx] = a
    t_elapsed = time.time() - t_start
    print('Review %d (%.4fs)'% (ridx, t_elapsed))
    return analysis, y_hat

analyze_scoresf(input_tensor_trainf, scores_trainf, 97)
analyze_scoress(input_tensor_trains, scores_trains, 97)

def plot_text_heatmap(words, scores, title="", width=5, height=0.2, verbose=0, max_word_per_line=10):
    fig = plt.figure(figsize=(width, height))
    
    ax = plt.gca()

    ax.set_title(title, loc='left')
    tokens = words
    if verbose > 0:
        print('len words : %d | len scores : %d' % (len(words), len(scores)))

    cmap = plt.cm.ScalarMappable(cmap=cm.bwr)
    cmap.set_clim(0, 1)
    
    canvas = ax.figure.canvas
    t = ax.transData

    # normalize scores to the followings:
    # - negative scores in [0, 0.5]
    # - positive scores in (0.5, 1]
    normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5
    
    if verbose > 1:
        print('Raw score')
        print(scores)
        print('Normalized score')
        print(normalized_scores)

    # make sure the heatmap doesn't overlap with the title
    loc_y = -0.2

    for i, token in enumerate(tokens):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)
        
        text = ax.text(0.0, loc_y, token,
                       bbox={
                           'facecolor': color,
                           'pad': 5.0,
                           'linewidth': 1,
                           'boxstyle': 'round,pad=0.5'
                       }, transform=t)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        
        # create a new line if the line exceeds the length
        if (i+1) % max_word_per_line == 0:
            loc_y = loc_y -  2.5
            t = ax.transData
        else:
            t = transforms.offset_copy(text._transform, x=ex.width+15, units='dots')

    if verbose == 0:
        ax.axis('off')

af, y_predf = analyze_scoresf(input_tensor_trainf, scores_trainf, 100)
ass, y_preds = analyze_scoress(input_tensor_trains, scores_trains, 100)

print(af[0][0])
print(ass[0][0])

plot_text_heatmap(
    texts_train[100],
    af[0][0]
)

plot_text_heatmap(
    texts_train[100],
    ass[0][0]
)

idx = 0
wordsf = texts_valf[idx]
wordss = texts_vals[idx]
    
print('Review(id=%d): %s' % (idx, ' '.join(wordsf)))
y_truef = scores_valf[idx]
af, y_predf = analyze_scoresf(input_tensor_valf, scores_valf, idx)

print("Pred class : %d %s" %
      (y_predf, '✓' if y_predf == y_truef else '✗ (%d)' % y_truef)
      )
                            
for j, method in enumerate(methods):
    plot_text_heatmap(wordsf, af[j].reshape(-1), title='Method: %s' % method, verbose=0)
    plt.show()
    print()
    
print('Review(id=%d): %s' % (idx, ' '.join(wordss)))
y_trues = scores_vals[idx]
ass, y_preds = analyze_scoress(input_tensor_vals, scores_vals, idx)

print("Pred class : %d %s" %
      (y_preds, '✓' if y_preds == y_trues else '✗ (%d)' % y_trues)
      )

for j, method in enumerate(methods):
    plot_text_heatmap(wordss, ass[j].reshape(-1), title='Method: %s' % method, verbose=0)
    plt.show()
    print()





























