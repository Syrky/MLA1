import re
import nltk
import numpy
import pandas

nltk.download('stopwords')  #stopwords
nltk.download('punkt')
from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))

from nltk.tokenize import word_tokenize  #tokenize

from nltk.stem import WordNetLemmatizer  #lemmanize
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.feature_extraction.text import TfidfVectorizer  #vectorize
vectorizer = TfidfVectorizer()


def delete_trash_from_txt_file(f_read):
    # txt_name = str(txt_name)
    # openfile = r"text_" + txt_name + ".txt"
    # reader = open(openfile)
    # f_read = reader.read()
    f_read = f_read.lower()  # Letters to lowercase

    f_read = re.sub(r"\d+", "", f_read, flags=re.UNICODE)  #deleting numbers
    f_read = re.sub(r"[^\w\s]", "", f_read, flags=re.UNICODE)  #deleting punctuation
    text_tokens = word_tokenize(f_read)  # tokenization
    f_read = [word
         for word in text_tokens
         if not word in en_stops]  # deleting stop words
    return f_read
    lemmatizer = WordNetLemmatizer()
    for i in range(len(f_read)):
        f_read[i] = lemmatizer.lemmatize(f_read[i])
    return(f_read)
  
    # df = pandas.value_counts(numpy.array(r))
    # for i in range(len(df.values)):
    #     freq = df. / valuesamount_of_words
    # print(str(df.index) + " " + str(freq))
    # for i in range(len(df.values)):
    #     print(df.index[i])
    # print(amount_of_words)
    # print(r)

def comp_tf(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def comp_idf(documents):
    import math
    N = len(documents)

    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def comp_TFxIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return(tfidf)

def prepare_file():
    with open('text_1.txt') as fr: file1 = fr.read()
    with open('text_2.txt') as fr: file2 = fr.read()
    with open('text_3.txt') as fr: file3 = fr.read()
    with open('text_4.txt') as fr: file4 = fr.read()
    with open('text_5.txt') as fr: file5 = fr.read()
    with open('text_6.txt') as fr: file6 = fr.read()
    with open('text_7.txt') as fr: file7 = fr.read()
    with open('text_8.txt') as fr: file8= fr.read()
    with open('text_9.txt') as fr: file9 = fr.read()
    with open('text_10.txt') as fr: file10 = fr.read()
    
    file1 = delete_trash_from_txt_file(file1)
    file2 = delete_trash_from_txt_file(file2)
    file3 = delete_trash_from_txt_file(file3)
    file4 = delete_trash_from_txt_file(file4)
    file5 = delete_trash_from_txt_file(file5)
    file6 = delete_trash_from_txt_file(file6)
    file7 = delete_trash_from_txt_file(file7)
    file8 = delete_trash_from_txt_file(file8)
    file9 = delete_trash_from_txt_file(file9)
    file10 = delete_trash_from_txt_file(file10)
    combined_file = set(file1).union(set(file2)).union(set(file3)).union(set(file4)).union(set(file5)).union(set(file6)).union(set(file7)).union(set(file8)).union(set(file9)).union(set(file10))
    
    wordDict1 = dict.fromkeys(combined_file, 0)
    wordDict2 = dict.fromkeys(combined_file, 0)
    wordDict3 = dict.fromkeys(combined_file, 0)
    wordDict4 = dict.fromkeys(combined_file, 0)
    wordDict5 = dict.fromkeys(combined_file, 0)
    wordDict6 = dict.fromkeys(combined_file, 0)
    wordDict7 = dict.fromkeys(combined_file, 0)
    wordDict8 = dict.fromkeys(combined_file, 0)
    wordDict9 = dict.fromkeys(combined_file, 0)
    wordDict10 = dict.fromkeys(combined_file, 0)
    
    for word in file1:  wordDict1[word] += 1
    for word in file2: wordDict2[word] += 1
    for word in file3: wordDict3[word] += 1
    for word in file4: wordDict4[word] += 1
    for word in file5: wordDict5[word] += 1
    for word in file6: wordDict6[word] += 1
    for word in file7: wordDict7[word] += 1
    for word in file8: wordDict8[word] += 1
    for word in file9: wordDict9[word] += 1
    for word in file10: wordDict10[word] += 1

    tf1 = comp_tf(wordDict1, file1)
    tf2 = comp_tf(wordDict2, file2)
    tf3 = comp_tf(wordDict3, file3)
    tf4 = comp_tf(wordDict4, file4)
    tf5 = comp_tf(wordDict5, file5)
    tf6 = comp_tf(wordDict6, file6)
    tf7 = comp_tf(wordDict7, file7)
    tf8 = comp_tf(wordDict8, file8)
    tf9 = comp_tf(wordDict9, file9)
    tf10 = comp_tf(wordDict10, file10)
    tf = pandas.DataFrame([tf1, tf2, tf3, tf4, tf5, tf6, tf7, tf8, tf9, tf10])
    File1 = open('tf_result.txt', 'a')
    File1.write(str(tf))
    File1.close()

    print("TF:")
    print(tf)
    print("")
    idfs = comp_idf(
        [wordDict1, wordDict2, wordDict3, wordDict4, wordDict5, wordDict6, wordDict7, wordDict8, wordDict9, wordDict10])
    File2 = open('idf_result.txt', 'a')
    File2.write(str(idfs))
    File2.close()
    print("IDFS:")
    print(idfs)
    print("")
    idf1 = comp_TFxIDF(tf1, idfs)
    idf2 = comp_TFxIDF(tf2, idfs)
    idf3 = comp_TFxIDF(tf3, idfs)
    idf4 = comp_TFxIDF(tf4, idfs)
    idf5 = comp_TFxIDF(tf5, idfs)
    idf6 = comp_TFxIDF(tf6, idfs)
    idf7 = comp_TFxIDF(tf7, idfs)
    idf8 = comp_TFxIDF(tf8, idfs)
    idf9 = comp_TFxIDF(tf9, idfs)
    idf10 = comp_TFxIDF(tf10, idfs)
    # putting it in a textframe
    TFxIDF = pandas.DataFrame([idf1, idf2, idf3, idf4, idf5, idf6, idf7, idf8, idf9, idf10])
    print("TFxIDF:")
    print(TFxIDF)
    File3 = open('tf-idf_result.txt', 'a')
    File3.write(str(TFxIDF))
    File3.close()
        
if __name__ == '__main__':
    prepare_file()