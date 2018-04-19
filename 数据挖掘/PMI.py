"""
jieba分词, 抽取文章的关键词，合并计算生成类里面的关键词, 计算两辆关键词之间的PMI。
根据PMI排序，判断处理后，计算出两个类别之间的连接词 。

py中涉及三个类别文章，简记a类，b类，ab混合后的类
docs_path: ab混合类文章，分词后的路径
"""

import os
import jieba
import jieba.analyse
import xlrd
import codecs
import math
import numpy as np
import pandas as pd


def fetch_text_from_xls(file_path):
    """
    从xls文件中获取文本
    """
    xls_data = xlrd.open_workbook(file_path)

    table_data = xls_data.sheet_by_name("热点数据")
    table_rows = table_data.nrows

    for i in range(1, table_rows):
        row_data = table_data.row_values(i)
        text_content = row_data[2]
        print(text_content)

        with codecs.open("/Users/fine_day/PycharmProjects/work/word_pmi/corpus/all/%s" % ("content_a"+str(i)+".txt"),
                         'a', encoding='utf-8') as f_content:
            f_content.write(text_content)

class Reader(object):

    def __init__(self, root_path):
        self.__root_path = root_path

    def cut_words(self):
        """
        分词，并将分词结果存在相应目录
        """
        # 第一层目录
        first_class_dir = os.listdir(self.__root_path)
        for file_full_name in first_class_dir:
            file_path = os.path.join(self.__root_path, file_full_name)

            try:
                print(file_path)
                if os.path.splitext(file_path)[1] == '.txt':
                    file = open(file_path, 'r')
                    file_content = file.read()
                    print(file_content)
                    # 去除'\n'
                    file_content_s = file_content.strip('\n').strip()
                    file_words_cut = jieba.cut(file_content_s)
                    file_words_cut_s = " ".join(file_words_cut)
                    print(file_words_cut_s)
                    file.close()

                    with codecs.open(r'/Users/fine_day/PycharmProjects/work/word_pmi/corpus/all_s/%s' % file_full_name, 'a',
                                     encoding='utf-8') as f:
                        f.write(file_words_cut_s)

            except Exception as e:
                print(e)

    def get_keywords(self):
        """
        用text-rank方法抽取关键词
        :return: 返回某类里面所有文章关键词的交集，每篇文章关键词20个
        """
        keywords_list = []
        first_class_dir = os.listdir(self.__root_path)
        for file_full_name in first_class_dir:
            file_path = os.path.join(self.__root_path, file_full_name)

            try:
                print(file_path)
                if os.path.splitext(file_path)[1] == '.txt':
                    file = open(file_path, 'r')
                    file_content = file.read()
                    # print(file_content)
                    # 去除'\n'
                    file_content_s = file_content.strip('\n').strip()
                    for x in jieba.analyse.textrank(file_content_s, topK=20, withWeight=False, allowPOS=("ns", "n",
                                                                                                         "vn")):
                        keywords_list.append(x)

                    file.close()

            except Exception as e:
                print(e)

        # keywords_list_duplicate = list(set(keywords_list))
        return keywords_list

    @staticmethod
    def write_words(word_list, file_path):
        # 写入操作和类或实例关系不大
        for word in word_list:
            with codecs.open(file_path, 'a', encoding='utf-8') as f:
                f.write(word + "\n")


def document_frequency(docs_path, word):
    """
    计算某个词出现的文档频率

    :param docs_path: 文档集合的路径
    :param word: 词
    :return: 某个词的文档频率
    """
    first_class_dir = os.listdir(docs_path)
    word_n = 0
    for file_full_name in first_class_dir:
        file_path = os.path.join(docs_path, file_full_name)
        try:
            # print(file_path)
            file = open(file_path, 'r')
            file_content = file.read()
            if word in file_content:
                word_n = word_n + 1
            file.close()

        except Exception as e:
            print(e)

    # print(word_n)
    word_p = word_n/(len(first_class_dir))
    return word_p


def document_frequency_tw(docs_path, word1, word2):
    """
    计算两个词同时出现的文档频率
    :param docs_path: 文档集合的路径
    :param word1: 词
    :param word2: 词
    :return: 返回词的文档频率
    """
    first_class_dir = os.listdir(docs_path)
    words_n = 0
    for file_full_name in first_class_dir:
        file_path = os.path.join(docs_path, file_full_name)
        try:
            # print(file_path)
            file = open(file_path, 'r')
            file_content = file.read()
            if word1 and word2 in file_content:
                words_n = words_n + 1
            file.close()

        except Exception as e:
            print(e)
    words_p = words_n/(len(first_class_dir))
    return words_p


def pmi(word1, word2):
    """
    计算连个关键词的PMI(关联度)
    :param word1:
    :param word2:
    :return: 返回两个词以及pmi值
    """
    docs_path = "/Users/fine_day/PycharmProjects/work/word_pmi/corpus/all_s"
    word1_p = document_frequency(docs_path, word1)
    word2_p = document_frequency(docs_path, word2)
    word1_word2_p = document_frequency_tw(docs_path, word1, word2)
    middle_value = word1_word2_p/(word1_p*word2_p)
    pmi_value = math.log(middle_value)
    # print(word1, word2, pmi_value)
    return pmi_value


def get_words_couple(word_list):
    """
    从列表中取两个词，取出所有的情况
    :param word_list: 词列表
    :return: 返回二维数组
    """
    word_couple_list = []
    for i in range(len(word_list)):
        word1 = word_list[i]
        for n in range(i+1, len(word_list)):
            word2 = word_list[n]
            word_couple = [word1, word2]
            word_couple_list.append(word_couple)

    return word_couple_list


def get_topic_keywords(keywords_list):
    """
    从关键词列表中，统计出词频最高的前三十个。
    :param keywords_list: 未去重的关键词列表
    :return: 返回前三十个词作为小类的关键词
    """
    keywords_duplicated_list = list(set(keywords_list))
    kw_num_list = []
    for keywords in keywords_duplicated_list:
        keywords_num = keywords_list.count(keywords)
        kw_num = [keywords, keywords_num]
        kw_num_list.append(kw_num)
    # kw_num_array = np.array(kw_num_list)
    kw_num_df = pd.DataFrame(kw_num_list)
    kw_num_df.columns = ['word', 'word_num']
    kw_num_df[['word_num']] = kw_num_df[['word_num']].astype(int)
    kw_sort = kw_num_df.sort_values(by="word_num", ascending=False).head(30)[['word']]

    return [kw for kw in kw_sort['word']]


def sub_list(a_list, b_list):
    """
    判断a_list是否是b_list的子集，或两者相等
    :param a_list:
    :param b_list:
    :return: 如果是返回True, 否则返回False
    """

    for el in a_list:
        if el in b_list and a_list.count(el) <= b_list.count(el):
            continue
        else:
            return False
    return True


def main():
    """
    reader_a,为对a类文章进行操作的Reader()实例，路径为a类文章（未分词）路径
    reader_b,为对b类文章进行操作的Reader()实例，路径为b类文章（未分词）路径
    """
    reader_a = Reader("/Users/fine_day/PycharmProjects/work/word_pmi/corpus/a")
    reader_b = Reader("/Users/fine_day/PycharmProjects/work/word_pmi/corpus/b")
    keywords_list_a = reader_a.get_keywords()
    keywords_list_b = reader_b.get_keywords()
    topic_kw_list_a = get_topic_keywords(keywords_list_a)
    topic_kw_list_b = get_topic_keywords(keywords_list_b)
    topic_kw_list_ab = topic_kw_list_a + topic_kw_list_b
    topic_kw_duplicated_list_ab = list(set(topic_kw_list_ab))
    word_couple_list = get_words_couple(topic_kw_duplicated_list_ab)
    # print(len(word_couple_list))
    pmi_two_dimensional_array = []
    for word_list in word_couple_list:
        word1 = word_list[0]
        word2 = word_list[1]
        pmi_value = pmi(word1, word2)
        pmi_list = [word1, word2, pmi_value]
        pmi_two_dimensional_array.append(pmi_list)
        # print(pmi_list)
    pmi_df = pd.DataFrame(pmi_two_dimensional_array)
    pmi_df.columns = ['word1', 'word2', 'pmi_value']
    pmi_df[['pmi_value']] = pmi_df[['pmi_value']].astype(float)
    pmi_df_sort = pmi_df.sort_values(by="pmi_value", ascending=False).head(100)
    print(pmi_df_sort)
    # print(pmi_df_sort[['word1', 'word2']])

    # 将排序好的pmi值前100 dataframe结构装换成二维的list
    pmi_sort_two_dimensional_array = np.array(pmi_df_sort[['word1', 'word2']]).tolist()
    pmi_sort_num = len(pmi_sort_two_dimensional_array)
    # pmi_couple_thd_array 为三维的list
    pmi_couple_thd_array = []
    for i in range(pmi_sort_num):
        pmi_couple1 = pmi_sort_two_dimensional_array[i]
        for n in range(i+1, pmi_sort_num):
            pmi_couple2 = pmi_sort_two_dimensional_array[n]
            pmi_couple_list = [pmi_couple1, pmi_couple2]
            pmi_couple_thd_array.append(pmi_couple_list)

    # pmi_couple_thd_array 的样式[[[word1, word2],[word3, word4]], [[],[]], ...]
    connect_word_list = []
    for word_td_array in pmi_couple_thd_array:
        word_ndarray1 = np.array(word_td_array[0])
        word_ndarray2 = np.array(word_td_array[1])
        connect_word = np.intersect1d(word_ndarray1, word_ndarray2)
        if connect_word:
            word_diff_array = np.setxor1d(word_ndarray1, word_ndarray2)
            word_diff_list = word_diff_array.tolist()
            if sub_list(word_diff_list, topic_kw_list_a) or sub_list(word_diff_list, topic_kw_list_b):
                pass
            else:
                connect_word_list.append((connect_word.tolist())[0])
    # connect_word_list_duplicated: 两类文章之间的连接词
    connect_word_list_duplicated = list(set(connect_word_list))
    print(connect_word_list_duplicated)

if __name__ == '__main__':
    main()

