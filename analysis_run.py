from datamanager import DataManager
from news_analysis import InputData
from gensim.models import Word2Vec
from news_analysis import LDABuilder
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def main():
    input = InputData()
    # builder = LDABuilder()
    # builder.main()


if __name__=='__main__':
    main()