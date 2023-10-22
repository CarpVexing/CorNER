
from nltk import data
data.path.append(r'/root/nltk_data')
import re
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import stem
from nltk.stem.wordnet import WordNetLemmatizer
import csv


def read_data(file):
    with open(file, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        lines=lines[1:]
    return [re.sub('[^0-9a-zA-Z\_\-]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines): #分词
    return [line.split() for line in lines]

def remove_stop_words(tokens,lines,stop_words,nostop_word_tokens):
    for i in range(len((lines))):
        word_list = tokens[i]
        word_lists = []
        for wd in word_list:
            if wd not in stop_words:
                 word_lists.append(wd)
        nostop_word_tokens.append(word_lists)
    return nostop_word_tokens

def get_pos(lines,nostop_word_tokens,nostop_word_pos_tokens):
    for i in range(len(lines)):
        nostop_word_pos_tokens.append(pos_tag(nostop_word_tokens[i]))
    return nostop_word_pos_tokens

# line_s = read_data()

# lines = []
# nostop_word_tokens = []
# nostop_word_pos_tokens = []
# nostop_word_pos_tokens_with = []#带空格的字符列表

# for line in line_s:
#     lines.append(line+' .')

# tokens = tokenize(lines)

# stop_words = stopwords.words('english')#去停用词

# nostop_word_tokens = remove_stop_words(tokens)

# nostop_word_pos_tokens = get_pos()

# lens = len(lines)

# for i in range(lens):
#     vec = []
#     for j in range(len(nostop_word_pos_tokens[i])):
#         if j == 0:
#             vec.append(('Sentence:%d' % (i+1), nostop_word_pos_tokens[i][j][0], nostop_word_pos_tokens[i][j][1]))
#         else:
#             vec.append(('', nostop_word_pos_tokens[i][j][0], nostop_word_pos_tokens[i][j][1]))
#     nostop_word_pos_tokens_with.append(vec)

# with open("./TSE/dataAll/spark/spark_sentence1_tokenize.csv", 'w', newline='') as f:
#     data_writer = csv.writer(f, lineterminator='\n')
#     header=['Sentence #','Word','POS']
#     data_writer.writerow(header)
#     for i in range(lens):
#         for wd in nostop_word_pos_tokens_with[i]:
#             data_writer.writerow(wd)
