import re
from konlpy.tag import Mecab

data = 'itc 단조강 수입으로 피해상무부 덤핑 조사 예정'
mecab = Mecab()
print(mecab.morphs(phrase=data))