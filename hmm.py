import pickle
import numpy as np
import string

class HMM():

    def __init__(self, file_name=None):
        self.file_name = file_name
        self.unigrams = {}
        self.tags = []
        self.bigrams = {}
        self.trigrams = {}
        self.word_tags = {}
        self.N = 0
        self.__PUNCT__ = list(string.punctuation)

    def __has_digit__(self, token):
        if len(token):
            for i in token:
                if i.isdigit():
                    return True
        return False

    def __is_punct__(self, token):
        if token in self.__PUNCT__:
            return True
        return False

    def parse_file(self):
        DICT = []
        f = open(self.file_name, 'r')
        sentences = []
        sent = [('*', '*'), ('*', '*')]
        for line in f:
            if line.strip():
                line = line.strip().split()
                if line[1] not in self.tags:
                    self.tags.append(line[1])
                sent.append((line[0].lower(), line[1]))
                if line[1] == 'M':
                    line[0] = 'is_number'
                    if line[0] not in DICT:
                        DICT.append(line[0])
                        self.N += 1
                else:
                    if line[0].lower() not in DICT:
                        DICT.append(line[0].lower())
                        self.N += 1
                if line[1] == 'CH':
                    if line[0] not in self.__PUNCT__:
                        self.__PUNCT__.append(line[0])
                word_tag = (
                    line[0].lower(),
                    line[1]
                )
                if word_tag not in self.word_tags:
                    self.word_tags[word_tag] = 1
                else:
                    self.word_tags[word_tag] += 1
            else:
                sent.append(('STOP', 'STOP'))
                sentences.append(sent)
                sent = [('*', '*'), ('*', '*')]
        f.close()
        return sentences

    def load(self, model_name):
        model = pickle.load(open(model_name, 'rb'))
        self.unigrams = model.unigrams
        self.bigrams = model.bigrams
        self.trigrams = model.trigrams
        self.tags = model.tags
        self.word_tags = model.word_tags
        self.N = model.N

    def set_ngrams(self):
        sentences = self.parse_file()
        for sent in sentences:
            for i in range(2, len(sent)):
                word_trigrams = (
                    sent[i-2][-1],
                    sent[i-1][-1],
                    sent[i][-1]
                )
                word_bigrams = (
                    sent[i-1][-1],
                    sent[i][-1]
                )
                if word_trigrams not in self.trigrams:
                    self.trigrams[word_trigrams] = 1
                else:
                    self.trigrams[word_trigrams] += 1
                if word_bigrams not in self.bigrams:
                    self.bigrams[word_bigrams] = 1
                else:
                    self.bigrams[word_bigrams] += 1
                if sent[i][-1] not in self.unigrams:
                    self.unigrams[sent[i][-1]] = 1
                else:
                    self.unigrams[sent[i][-1]] += 1
        self.bigrams[('*', '*')] = 4504
    def get_emission(self, word, tag):
        word_tag = (
            word,
            tag
        )
        if word_tag in self.word_tags:
            return float(self.word_tags[word_tag] + 1.0)/float(self.unigrams[tag])
        else:
            return 1.0/float(self.unigrams[tag])

    def get_transition(self, first, second, third):
        word_trigram = (
            first,
            second,
            third
        )
        word_bigram = (
            first,
            second
        )
        if word_trigram in self.trigrams:
            return float(self.trigrams[word_trigram] + 1.0)/float(self.bigrams[word_bigram] + self.N)
        else:
            if word_bigram in self.bigrams:
                return 1.0/float(self.bigrams[word_bigram] + self.N)
            else:
                return 0.0

    def get_transition2(self, first, second):
        word_bigrams = (
            first,
            second
        )
        print(word_bigrams)
        if word_bigrams in self.bigrams[word_bigrams]:
            return float(self.bigrams[word_bigrams] + 1.0)/float(self.unigrams[first] + self.N)
        else:
            return 1.0/float(self.unigrams[first])


    def preprocess(self, tokens):
        for token in tokens:
            if self.__has_digit__(token):
                token = 'is_number'
        return tokens

    def viterbi(self, tokens):
        tokens = self.preprocess(tokens)
        self.__PUNCT__ = list(set(self.__PUNCT__))
        answertag = []
        len1 = len(tokens) + 1
        len2 = len(self.tags) + 1
        viterbi_prob = np.zeros((len1, len2, len2))
        backpointer = np.empty((len1, len2, len2), dtype=int)
        for p in range(len(self.tags)):
            tag = self.tags[p]
            emission = self.get_emission(tokens[0], tag)
            transition = self.get_transition('*', '*', tag)
            viterbi_prob[1, 0, p] = emission*transition
        if len(tokens) > 1:
            for q in range(len(self.tags)):
                tag = self.tags[q]
                emission = self.get_emission(tokens[1], tag)
                for p in range(len(self.tags)):
                    prev_tag = self.tags[p]
                    transition = self.get_transition('*', prev_tag, tag)
                    viterbi_prob[2, p, q] = viterbi_prob[1, 0, p]*emission*transition
        else:
            return []
        if len(tokens) > 2:
            for t in range(3, len(tokens) + 1):
                for q in range(len(self.tags)):
                    tag = self.tags[q]
                    for p in range(len(self.tags)):
                        prev_tag = self.tags[p]
                        maxscore = 0.0
                        maxindex = -1
                        for w in range(len(self.tags)):
                            prev_2_tag = self.tags[w]
                            emission = self.get_emission(tokens[t-1], tag)
                            transition = self.get_transition(prev_2_tag, prev_tag, tag)
                            current = viterbi_prob[t-1, w, p]*transition*emission
                            if current > maxscore:
                                maxscore = current
                                maxindex = w
                        viterbi_prob[t, p, q] = maxscore
                        backpointer[t, p, q] = maxindex
        maxscore = -1.0
        id_1 = None
        id_2 = None
        if len(tokens) <= 2:
            if len(tokens) == 1:
                for p in range(len(self.tags)):
                    transition = self.get_transition('*', self.tags[p], 'STOP')
                    current = viterbi_prob[1, 0, p]*transition
                    if current > maxscore:
                        maxscore = current
                        id_1 = p
                return [self.tags[id_1]]
            else:
                for p in range(len(self.tags)):
                    for w in range(len(self.tags)):
                        transition = self.get_transition(self.tags[w], self.tags[p], 'STOP')
                        current = viterbi_prob[2, w, p]*transition
                        if current > maxscore:
                            maxscore = current
                            id_1 = p
                            id_2 = w
                return [self.tags[id_2], self.tags[id_1]]
        for q in range(len(self.tags)):
            prev_tag = self.tags[q]
            for p in range(len(self.tags)):
                prev_2_tag = self.tags[p]
                transition = self.get_transition(prev_2_tag, prev_tag, 'STOP')
                if tokens[-1] not in self.__PUNCT__:
                    if prev_tag == 'CH':
                        transition = 0.0
                current = viterbi_prob[len(tokens), p, q] * transition
                if current > maxscore:
                    maxscore = current
                    id_2 = p
                    id_1 = q
        answertag.append(self.tags[id_1])
        answertag.append(self.tags[id_2])
        for t in range(len(tokens), 2, -1):
            temp = id_2
            id_2 = backpointer[t, id_2, id_1]
            answertag.append(self.tags[id_2])
            id_1 = temp
        answertag.reverse()
        return answertag



if __name__ == '__main__':
    model = HMM('vlsp/dev.txt')
    model.load('hmm.pkl')
    model.set_ngrams()
    sentences = []
    tags = []
    f = open('vlsp/test.txt', 'r')
    sent, tag = [], []
    for line in f:
        if line.strip():
            line = line.strip().split()
            if line[-1] == 'M':
                line[0] = "is_number"
            sent.append(line[0].lower())
            tag.append(line[-1])
        else:
            sentences.append(sent)
            tags.append(tag)
            sent = []
            tag = []
    f.close()
    count = 0
    total = 0
    out = open('test.txt', 'w')
    for i, sent in enumerate(sentences):
        y_pred = model.viterbi(sent)
        for j, tag in enumerate(y_pred):
            if tag == tags[i][j]:
                count += 1
            total += 1
            out.write(sent[j] + '\t' + tags[i][j] + '\t' + tag)
            out.write('\n')
        out.write('\n')
    print('Acc: %.2f' % (count*100/total), end=' %')
