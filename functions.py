import pickle
from keras.models import load_model
import numpy as np


def translate_sentence(input_):
    encoder_model = load_model('french_encoder_models.h5')
    decoder_model = load_model('decoder_model.h5')
    word2idx_outputs = pickle.load(open('word2idx_outputs', 'rb'))
    word2idx_inputs = pickle.load(open("word2idx_inputs", 'rb'))
    idx2word_target = pickle.load(open("idx2word_target", 'rb'))
    list_ = [(word2idx_inputs[word]) for word in input_.split(' ')]
    if len(list_) > 5:
        print("number of worlds in input must be less than 6")
        return
    for i in range(0, 5 - len(list_)):
        list_.insert(i, 0)
    input_seq = np.array([list_])
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(12):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)
