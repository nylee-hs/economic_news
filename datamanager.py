import pandas as pd
import csv
from tqdm import tqdm
import re
# from konlpy.tag import Mecab
import pymysql
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()
import MySQLdb

class DataManager:
    def __init__(self):
        self.host = input('host ip : ')
        self.user = input('user id : ')
        self.password = input('password : ')
        self.port = 3306
        self.db = 'economic_database'

    def load_csv(self, file, encoding, data_types):
        csv_data = pd.read_csv(file, encoding=encoding, dtype=data_types)
        return csv_data

    def save_csv(dataFrame, file_name, save_mode):
        dataFrame.to_csv(file_name, mode=save_mode, encoding='ms949')
        print('파일 저장 완료')

    def text_cleanning(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#         u"\U0001F1F2-\U0001F1F4"  # Macau flag
#         u"\U0001F1E6-\U0001F1FF"  # flags
#         u"\U0001F600-\U0001F64F"
#         u"\U00002702-\U000027B0"
#         u"\U000024C2-\U0001F251"
#         u"\U0001f926-\U0001f937"
#         u"\U0001F1F2"
#         u"\U0001F1F4"
#         u"\U0001F620"
#         u"\u200d"
#         u"\u2640-\u2642"
        "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        text = re.sub('<.+?>', '', text, 0).strip()
        text = re.sub('[^0-9a-zA-Zㄱ-힗]', ' ', text)
        text = ' '.join(text.split())
        pattern_email = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9]+\.[a-zA-Z0-9.]+)'
        repl = ''
        text = re.sub(pattern=pattern_email, repl=repl, string=text)
        text = re.sub('[-=+,#/\?:^$.@*\"”※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', repl, text)
        text = text.replace('·', '')
        text = text.replace('“', '')
        text = text.replace('”', '')
        return text

    # def dataMorph(file_name):
    #     nouns_list = []
    #     csv_data = DataManager.load_csv(file_name, "text")
    #     mecab = Mecab()
    #     nouns_list = [mecab.nouns(text) for text in csv_data['text']]
    #     csv_data['nouns'] = nouns_list
    #     save_csv(csv_data, file_name, "w")
    #     return csv_data

    def getData_fromCSV(file_name):
        csv_data = DataManager.load_csv(file_name)
        return csv_data

    def connectionDB(self):
        conn = pymysql.connect(host=self.host, user=self.user, password=self.password, port=self.port, db=self.db, charset='utf8')
        return conn

    def select_all_db(self, table):
        conn = self.connectionDB()
        sql = f'select * from {table}'
        result = pd.read_sql_query(sql, conn)
        print(result.head())
        conn.close()
        return result

    def insert_db(self, table, data):
        engine = create_engine(f'mysql+mysqldb://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}', encoding='utf-8')
        conn = engine.connect()
        data.to_sql(name=table, con=conn, if_exists='append', index=False)
        conn.close()

    # def get_nouns(self, data):


# if __name__=='__main__':
#     dm = DataManager()
#     dm.host = input('host ip : ')
#     dm.user = input('user id : ')
#     dm.password = input('password : ')
#
#     data = dm.select_all_db('eco_news_data')


    # data_types = {'뉴스 식별자':str}
    # data = dm.load_csv(file='data/NewsResult_import_final.csv', encoding='utf-8', data_types = data_types)
    # data['뉴스 식별자'] = data['뉴스 식별자'].astype(str)
    # data = data.loc[:, ~data.columns.str.match('Unnamed')]
    # data.columns=['뉴스식별자','일자','언론사','기고자','제목','통합분류1','통합분류2','통합분류3','사건_사고분류1','사건_사고분류2','사건_사고분류3','인물','위치','기관','키워드','특성추출','본문','URL','분석제외여부','추출명사']
    # print(data.tail())
    # dm.insert_db('eco_news_data', data)


    # result = dm.select_all_db('blog_data')
    # contents = data['Contents']
    # mecab = Mecab()
    # nouns_list = []
    # for text in contents:
    #     nouns = mecab.nouns(text)
    #     nouns = ' '.join(nouns)
    #     nouns_list.append(nouns)
    #
    # data['Nouns'] = nouns_list
    # # print(data['Nouns'][0])
    # dm.insert_db('blog_data_new', data)

    # dm.connectionDB()
    # conn = dm.connectionDB()
    # data = dm.load_csv('blog_data_20200514.csv', 'utf-8')
    #
    # dm.insert_db(data)
    # data.to_sql(name='blog_data.blog_data', con=conn, if_exists='append')
    # conn.close()





