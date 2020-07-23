import pickle
import os.path
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


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class InputData:
    def __init__(self):
        self.corpus = self.pre_prosseccing()
        self.model_path = 'models/0722/'
        self.word_count, self.word_count_list = self.get_word_count()

    def make_bigram(self, text, trigram_check):  ## trigram_check == 1 --> trigram, 0 --> bigram
        # min_count : Ignore all words and bigrams with total collected count lower than this value.
        # threshold : Represent a score threshold for forming the phrases (higher means fewer phrases).
        #             A phrase of words a followed by b is accepted if the score of the phrase is greater than threshold.
        #             Heavily depends on concrete scoring-function, see the scoring parameter.

        if trigram_check == 0:
            print('...make bigram...')
            bigram = gensim.models.Phrases(text, min_count=5, threshold=30.0)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            return [bigram_mod[doc] for doc in text]
        elif trigram_check == 1:
            print('...make trigram...')
            bigram = gensim.models.Phrases(text, min_count=5, threshold=20.0)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram = gensim.models.Phrases(bigram[text], threshold=20.0)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            return [trigram_mod[bigram_mod[doc]] for doc in text]

    def data_text_cleansing(self, data):
        print('Run text cleanning...')
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
            print('Stop Words File is found')
            dm = DataManager()
            df = dm.load_csv(file='data/doc2vec_test_data/0702/stopwords_list.csv', encoding='utf-8')
            stop_words_list = df['Stopwords'].tolist()
        else:
            print('Stop Words File is not found')
        return stop_words_list

    def get_including_words(self, path):
        file = 'including_words_list.csv'
        including_words_list = []
        if os.path.isfile(path+'/'+file):
            print('Including Words File is found')
            dm = DataManager()
            df = dm.load_csv(file=path+'including_words_list.csv', encoding='utf-8')
            including_words_list = df['Includingwords'].tolist()
        else:
            print('Including Words File is not found')
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
        print('Filtering words...')
        including_list = self.get_including_words('data/doc2vec_test_data/0702/')
        return [[word for word in simple_preprocess(str(doc)) if word in including_list] for doc in texts]

    def lematization(self, texts):
        print('Make lematization...')
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

        with open('models/0722/model.corpus', 'wb') as f:
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
        df.to_csv(self.model_path+'frequency.csv', 'w', 'utf-8')
        return word_count, word_count_list


class LDABuilder:
    def __init__(self):
        self.corpus = self.get_corpus('models/0722/model.corpus')
        self.num_topics = 30
        self.model_path = ''

    def get_corpus(self, corpus_file):
        with open(corpus_file, 'rb') as f:
            corpus = pickle.load(f)
        return corpus

    def getOptimalTopicNum(self):
        cntVec = CountVectorizer(min_df=2)
        nouns = [' '.join(arr) for arr in self.corpus]
        cntVec.fit(nouns)
        vec_matrix = cntVec.transform(nouns)
        features = cntVec.get_feature_names()
        corpus = Sparse2Corpus(vec_matrix, documents_columns=False)
        dictionary = Dictionary([features])

        com_nums = []
        for i in range(0, 100, 10):
            if i == 0:
                p = 1
            else:
                p = i
            com_nums.append(p)

        perplexity_list = []

        for i in com_nums:
            lda = LdaModel(corpus, i, dictionary, iterations=10, alpha='auto', random_state=0, passes=10)
            perp = lda.log_perplexity(corpus)
            perplexity_list.append(perp)
            print('k = {}  perplexity = {}'.format(str(i), str(perp)))

        perp_dict = dict(zip(com_nums, perplexity_list))
        sorted_perp_dict = sorted(perp_dict.items(), key=operator.itemgetter(1), reverse=True)

        plt.plot(com_nums, perplexity_list)
        plt.xlabel('topic')
        plt.ylabel('perplexity')
        plt.show()

        return sorted_perp_dict[0][0]

    def make_save_path(self, full_path):
        model_path = '/'.join(full_path.split('/')[:-1])
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return model_path

    def saveLDAModel(self, model_path):
        self.make_save_path(model_path)

        dictionary = corpora.Dictionary(self.corpus)
        corpus = [dictionary.doc2bow(text) for text in self.corpus]
        self.num_topics = self.getOptimalTopicNum()

        lda_model = ldamulticore.LdaMulticore(corpus, id2word=dictionary, num_topics=self.num_topics, workers=4)
        all_topics = lda_model.get_document_topics(corpus, minimum_probability=0.5, per_word_topics=False)

        documents = []
        with open(model_path + '.results', 'w', -1, 'utf-8') as f:
            for doc_idx, topic in enumerate(all_topics):
                if len(topic) == 1:
                    topic_id, prob = topic[0]
                    f.writelines(documents[doc_idx].strip() + "\u241E" + ' '.join(self.corpus[doc_idx]) + "\u241E" + str(topic_id) + "\u241E" + str(prob))

        lda_model.save(model_path + '.model')
        with open(model_path+'.dictionary', 'wb') as f:
            pickle.dump(dictionary, f)

        return lda_model

    def main(self):
        self.getOptimalTopicNum()
        self.model_path = self.make_save_path('models/0722')
        self.saveLDAModel(self.model_path)


if __name__ == '__main__':
    InputData()
    # dvi.pre_prosseccing()

# dvi = Doc2VecInput()
