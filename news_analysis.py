import numpy as np
import pickle
import os.path

import pyLDAvis
from tqdm import tqdm
from datamanager import DataManager
from gensim.models.doc2vec import TaggedDocument
import gensim
from gensim.models import Doc2Vec
from gensim.utils import simple_preprocess
import re
import pandas as pd
import logging
from konlpy.tag import Mecab
from sklearn.decomposition import TruncatedSVD
from soynlp.word import pmi
from soynlp.vectorizer import sent_to_word_contexts_matrix
from sklearn.feature_extraction.text import CountVectorizer
from gensim.matutils import Sparse2Corpus
from gensim.models import LdaModel
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
from gensim.models import ldamulticore
from gensim import corpora
import matplotlib.pyplot as plt
import operator
from gensim.models import CoherenceModel
import pyLDAvis.gensim as gensimvis

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class InputData:
    def __init__(self):
        self.model_path = self.make_save_path()
        self.corpus = self.pre_prosseccing()
        self.word_count, self.word_count_list = self.get_word_count()

    def make_save_path(self): ## directory는 'models/날짜'의 형식으로 설정해야 함
        print('==== Preprocessing ====')
        directory = input('model path : ')
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def make_bigram(self, text, trigram_check):  ## trigram_check == 1 --> trigram, 0 --> bigram
        # min_count : Ignore all words and bigrams with total collected count lower than this value.
        # threshold : Represent a score threshold for forming the phrases (higher means fewer phrases).
        #             A phrase of words a followed by b is accepted if the score of the phrase is greater than threshold.
        #             Heavily depends on concrete scoring-function, see the scoring parameter.

        if trigram_check == 0:
            print(' ...make bigram...')
            bigram = gensim.models.Phrases(text, min_count=5, threshold=30.0)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            return [bigram_mod[doc] for doc in text]
        elif trigram_check == 1:
            print(' ...make trigram...')
            bigram = gensim.models.Phrases(text, min_count=5, threshold=20.0)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram = gensim.models.Phrases(bigram[text], threshold=20.0)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            return [trigram_mod[bigram_mod[doc]] for doc in text]

    def data_text_cleansing(self, data):
        print(' ...Run text cleanning...')
        # Convert to list
        data = [re.sub('([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', '', str(sent)) for sent in data]
        # pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'  # URL제거
        data = [re.sub('(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', str(sent)) for sent in data]
        # pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
        data = [re.sub('([ㄱ-ㅎㅏ-ㅣ]+)', '', str(sent)) for sent in data]
        pattern = '<[^>]*>'  # HTML 태그 제거
        data = [re.sub(pattern=pattern, repl='', string=str(sent)) for sent in data]
        pattern = '[^\w\s]'  # 특수기호제거
        data = [re.sub(pattern=pattern, repl='', string=str(sent)) for sent in data]
        # data = data.tolist()

        # # 영문자 이외의 문자는 공백으로 변환
        # data = [re.sub('[^a-zA-Z]', ' ', str(sent)) for sent in data]
        #
        # for sent in data:
        #     print(sent)

        # Remove new line characters
        data = [re.sub('\s\s+', ' ', str(sent)) for sent in data]

        # Remove distracting single quotes
        data = [re.sub('\'', '', sent) for sent in data]

        return data

    def get_stop_words(self, path):
        file = 'stopwords_list.csv'
        stop_words_list = []
        if os.path.isfile(path+'/'+file):
            print('  ..Stop Words File is found..')
            dm = DataManager()
            df = dm.load_csv(file='data/doc2vec_test_data/0702/stopwords_list.csv', encoding='utf-8')
            stop_words_list = df['Stopwords'].tolist()
        else:
            print('  ..Stop Words File is not found..')
        return stop_words_list

    def get_including_words(self, path):
        file = 'including_words_list.csv'
        including_words_list = []
        if os.path.isfile(path+'/'+file):
            print('  ..Including Words File is found..')
            dm = DataManager()
            df = dm.load_csv(file=path+'including_words_list.csv', encoding='utf-8')
            including_words_list = df['Includingwords'].tolist()
        else:
            print('  ..Including Words File is not found..')
        print(including_words_list)
        return including_words_list

    # def remove_stopwords(self, texts):
    #     print('Remove stopwords...')
    #     stop_words = stopwords.words('english')
    #     stopwords_list = self.get_stop_words('data/doc2vec_test_data/0702')
    #     print('Append stopwords list: ', len(stopwords_list), 'words')
    #     # stop_words.extend(stopwords_list)  #추가할 stopwords list
    #     return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def word_filtering(self, texts):
        print(' ...Filtering words...')
        including_list = self.get_including_words('data/doc2vec_test_data/0702/')
        return [[word for word in simple_preprocess(str(doc)) if word in including_list] for doc in texts]

    def lematization(self, texts):
        print(' ...Make lematization...')
        mecab = Mecab()
        texts_out = []
        for sent in tqdm(texts):
            doc = " ".join(sent)
            texts_out.append(mecab.nouns(doc))
        # print(texts_out[0])
        return texts_out

    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

    def make_unique_words(self, data_lemmatized):
        print(data_lemmatized)
        uniquewords = [list(set(item)) for item in data_lemmatized]
        return uniquewords

    def pre_prosseccing(self):
        dm = DataManager()
        data = dm.select_all_db('eco_news_data')
        data = data['제목']
        with open(self.model_path + '/model.documents', 'wb') as f:
            pickle.dump(data, f)
        # # 수정된 job_title에서 posting_id 가지고 오기
        # posting_ids = data['posting_id']
        # posting_list = posting_ids.to_list()
        #
        # # posting_id에 따라 description_data set 만들기
        # des_data = [data['job_description'][id] for id in posting_ids]
        # title_data = [data['job_title'][id] for id in posting_ids]
        # id_list = [i for i in range(len(posting_list))]
        # df = pd.DataFrame({'id': posting_list, 'job_title': title_data, 'job_description': des_data, 'posting_id':posting_list})
        # df.to_csv('data/doc2vec_test_data/0702/merge_0629_adj.csv', mode='w', encoding='utf-8')

        # 수정된 description set 불러와 데이터 전처리 수행
        # data = dm.load_csv(file='data/doc2vec_test_data/0702/merge_0629_adj.csv', encoding='utf-8')
        sentences = self.data_text_cleansing(data)
        data_words = list(self.sent_to_words(sentences))
        # data_words_nostops = self.remove_stopwords(data_words)
        # data_lemmatized = self.lematization(data_words)
        # print(data_lemmatized)
        # bigram = self.make_bigram(data_lemmatized)


## 형태소 분석을 먼저 수행한 후 bigram을 만들어야 함
        data_lemmatized = self.lematization(data_words)
        trigram = self.make_bigram(data_lemmatized, trigram_check=1)

        with open(self.model_path + '/model.corpus', 'wb') as f:
            pickle.dump(trigram, f)
        return trigram

    def get_word_count(self):
        sline = [' '.join(line) for line in self.corpus]
        word_list = []
        for line in sline:
            for word in line.split():
                word_list.append(word)
        word_count = {}
        for word in word_list:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
        word_count_list = sorted(word_count.items(), key = lambda x:x[1], reverse=True)
        key_list = []
        value_list = []
        for item in word_count_list:
            key_list.append(item[0])
            value_list.append(item[1])
        df = pd.DataFrame({'Terms':key_list, 'Frequency': value_list})
        df.to_csv(self.model_path+'/frequency.csv', 'w', 'utf-8')
        return word_count, word_count_list


class LDABuilder:
    def __init__(self):
        self.model_path = self.make_save_path()
        self.corpus = self.get_corpus(self.model_path+ '/model.corpus')[:100]
        self.num_topics = self.getOptimalTopicNum()
        self.documents = self.get_documents(self.model_path + '/model.documents')[:100]

    def get_corpus(self, corpus_file):
        with open(corpus_file, 'rb') as f:
            corpus = pickle.load(f)
        return corpus

    def get_documents(self, documents_file):
        with open(documents_file, 'rb') as f:
            documents = pickle.load(f)
        return documents

    def getOptimalTopicNum(self):
        dictionary = corpora.Dictionary(self.corpus)
        corpus = [dictionary.doc2bow(text) for text in self.corpus]

        com_nums = []
        for i in range(0, 100, 10):
            if i == 0:
                p = 1
            else:
                p = i
            com_nums.append(p)

        coherence_list = []

        for i in com_nums:
            # lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
            #                                       id2word=dictionary,
            #                                       num_topics=i,
            #                                       iterations=100,
            #                                       alpha='auto',
            #                                       random_state=100,
            #                                       update_every=1,
            #                                       chunksize=10,
            #                                       passes=20,
            #                                       per_word_topics=True)
            lda = ldamulticore.LdaMulticore(corpus=corpus,
                                            id2word=dictionary,
                                            passes=20,
                                            num_topics=i,
                                            workers=4,
                                            iterations=100,
                                            alpha='symmetric',
                                            gamma_threshold=0.001)
            coh_model_lda = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
            coherence_value = coh_model_lda.get_coherence()

            # coh = lda.log_perplexity(corpus)
            coherence_list.append(coherence_value)
            print('k = {}  coherence value = {}'.format(str(i), str(coherence_value)))

        coh_dict = dict(zip(com_nums, coherence_list))
        sorted_coh_dict = sorted(coh_dict.items(), key=operator.itemgetter(1), reverse=True)

        plt.plot(com_nums, coherence_list)
        plt.xlabel('topic')
        plt.ylabel('coherence value')
        plt.draw()
        fig = plt.gcf()
        fig.savefig(self.model_path+'/coherence.png')
        t_ind = np.argmin(coherence_list)
        self.num_topics = t_ind * 10
        print('optimal topic number = ', str(t_ind))
        return sorted_coh_dict[0][0]

    def make_save_path(self): ## directory는 'models/날짜'의 형식으로 설정해야 함
        print('==== Modeling Building Process ====')
        directory = input('model path : ')
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def saveLDAModel(self, model_path):
        print(' ...start to build lda model...')
        dictionary = corpora.Dictionary(self.corpus)
        corpus = [dictionary.doc2bow(text) for text in self.corpus]

        # lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
        #                                           id2word=dictionary,
        #                                           num_topics=self.num_topics,
        #                                           iterations=100,
        #                                           alpha='auto',
        #                                           random_state=100,
        #                                           update_every=1,
        #                                           chunksize=10,
        #                                           passes=20,
        #                                           per_word_topics=True)

        lda_model = ldamulticore.LdaMulticore(corpus=corpus,
                                        id2word=dictionary,
                                        passes=20,
                                        num_topics=self.num_topics,
                                        workers=4,
                                        iterations=100,
                                        alpha='symmetric',
                                        gamma_threshold=0.001)

        all_topics = lda_model.get_document_topics(corpus, minimum_probability=0.5, per_word_topics=False)

        documents = self.documents
        with open(model_path + '/lda.results', 'w', -1, 'utf-8') as f:
            for doc_idx, topic in enumerate(all_topics):
                if len(topic) == 1:
                    topic_id, prob = topic[0]
                    f.writelines(documents[doc_idx].strip() + "\u241E" + ' '.join(self.corpus[doc_idx]) + "\u241E" + str(topic_id) + "\u241E" + str(prob) + '\n')
        lda_model.save(model_path + '/lda.model')
        with open(model_path+'model.dictionary', 'wb') as f:
            pickle.dump(dictionary, f)

        return lda_model

    def main(self):
        # self.model_path = self.make_save_path('models/0722')
        self.saveLDAModel(self.model_path)

class LDAModeler:
    def __init__(self):
        self.model_path = self.get_model_path()
        self.all_topics = self.load_results(self.model_path+'/lda.results')
        self.model = LdaModel.load(self.model_path+'/lda.model')
        self.corpus = self.get_corpus(self.model_path+'/model.corpus')
        self.dictionary = self.get_dictionary(self.model_path+'/lda.model.id2word')

    def get_model_path(self): ## directory는 'models/날짜'의 형식으로 설정해야 함
        print('==== LDA Model Analyzer ====')
        directory = input(' model path : ')
        return directory

    def view_lda_model(self, model, corpus, dictionary):
        prepared_data = gensimvis.prepare(model, corpus, dictionary)
        print(prepared_data)
        # pyLDAvis.save_html(prepared_data, self.model_path+'/vis_result.html')

    def get_corpus(self, corpus_file):
        with open(corpus_file, 'rb') as f:
            corpus = pickle.load(f)
        return corpus

    def load_results(self, result_fname):
        topic_dict = defaultdict(list)
        with open(result_fname, 'r', encoding='utf-8') as f:
            for line in f:
                sentence, _, topic_id, prob = line.strip().split('\u241E')
                topic_dict[int(topic_id)].append((sentence, float(prob)))

        for key in topic_dict.keys():
            topic_dict[key] = sorted(topic_dict[key], key=lambda x:x[1], reverse=True)
        return topic_dict

    def show_topic_docs(self, topic_id, topn=10):
        return self.all_topics[topic_id][:topn]

    def show_topic_words(self, topic_id, topn=10):
        return self.model.show_topic(topic_id, topn)

    def show_topics(self, model):
        return self.model.show_topics(fommated=False)

    def show_new_dociment_topic(self, documents, model):
        mecab = Mecab()
        tokenized_documents = [mecab.morphs(document) for document in documents]
        curr_corpus = [self.model.id2word.doc2bow(tokenized_documents) for tokenized_document in tokenized_documents]
        topics = self.model.get_document_topics(curr_corpus, minimum_probability=0.5, per_word_topics=False)
        for doc_idx, topic in enumerate(topics):
            if len(topic) == 1:
                topic_id, prob = topic[0]
                print(documents[doc_idx], ', topic id: ', str(topic_id), ', prob:', str(prob))
            else:
                print(documents[doc_idx], ', there is no dominant topic.')

    def get_dictionary(self, dic_fname):
        with open(dic_fname, 'rb') as f:
            dictionary = pickle.load(f)
        return dictionary

def main():
    builder = LDABuilder()
    builder.main()
    model = LDAModeler()

    for i in range(10):
        print(model.show_topic_words(i))
    model.view_lda_model(model.model, model.corpus, model.dictionary)
if __name__ == '__main__':
    main()




# dvi = Doc2VecInput()
