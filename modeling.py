import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models, similarities
import re
from pyvi.ViTokenizer import tokenize
import pickle
from func import word_analysis


def contentbased_filtering(df_text, method, unusing_words):
    df_text["Content_wt"]=df_text["Content_cleaned"].apply(lambda x: tokenize(x))
    match method:
        case 'cosine':
            print('Vào cosine')
            vectorizer = TfidfVectorizer(analyzer='word', stop_words=unusing_words)
            tfidf_matrix = vectorizer.fit_transform(df_text['Content_wt'])

            # Tính toán độ tương đồng
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            with open('models/cosine_sim.pkl', 'wb') as f:
                if pickle.dump(cosine_sim, f):
                    print('Mô hình cosine đã được lưu.')
            return cosine_sim
        case 'gemsim':
            # Tokenize(split) the sentences into words
            content_gem = [[text for text in x.split()] for x in df_text.Content_wt]
            # Tiền xử lý dữ liệu
            content_gem_re = [[re.sub('[0-9]+','', e) for e in text] for text in content_gem] # xem xét có cần bỏ các con số hay không
            content_gem_re = [[t.lower() for t in text if not t in ['', ' ', ',', '.', '...', '-',':', ';', '?', '%', '(', ')', '+', '/', "'", '&']] for text in  content_gem_re] # kiểm tra nội dung và đưa vào các ký tự đặc biệt
            content_gem_re = [[t for t in text if not t in unusing_words] for text in content_gem_re] # stopword
            dictionary = corpora.Dictionary(content_gem_re)
            dictionary.save('models/gensim.dict')   
            # Numbers of features (word) in dictionary
            feature_count = len(dictionary.token2id) 
            # Obtain corpus based on dictionary (dense matrix)
            corpus = [dictionary.doc2bow(text) for text in content_gem_re]
            #corpora.MmCorpus.serialize('/models/corpus.mm', corpus)
            # Use TF-IDF Model to process corpus, obtaining index
            tfidf = models.TfidfModel(corpus)
            tfidf.save("models/tfidf_model")
            print('Mô hình tfidf đã được lưu.')
            # tính toán sự tương tự trong ma trận thưa thớt
            index = similarities.SparseMatrixSimilarity(tfidf[corpus],
                                                        num_features = feature_count)
            if index.to_csv('models/gensim.csv'):
                print('Mô hình gensim đã được lưu.')
            return index
        
def cleaning_hotel_info(df_info):
    # Thay dấu , thành . để chuyển về kiểu float
    cols = ['Total_Score','Location','Cleanliness','Service','Facilities','Value_for_money','Comfort_and_room_quality']
    for col in cols:
        df_info[col] = df_info[col].replace(['', 'nan', 'No information'], np.nan)  # Thay chuỗi rỗng bằng NaN
        df_info[col] = df_info[col].astype(str).str.replace(',', '.', regex=False)
        df_info[col] = df_info[col].astype(float)
        df_info[col] = df_info[col].fillna(df_info[col].mean()).round(1)  # Thay NaN bằng giá trị trung bình của cột

    # Thay các giá trị không hợp lệ trong cột 'Hotel_Rank', 'Hotel_Description'
    df_info['Hotel_Rank'] = df_info['Hotel_Rank'].replace(['No information'], '0 sao trên 5') 
    df_info['Hotel_Description'] = df_info['Hotel_Description'].fillna(df_info['Hotel_Name']+'-'+df_info['Hotel_Address'])
    df_info.to_csv("models/cleaned_hotel_info.csv")

#>>> MAIN <<<
##DATA
file_info = 'data/hotel_info.csv'
file_comments = 'data/hotel_comments.csv'

##PROCESSING
df_info = pd.read_csv(file_info)
df_comments = pd.read_csv(file_comments)
### tiền xử lý
#cleaning_hotel_info(df_info)

#df_info_comments = word_analysis(df_info, df_comments) # chỉ chạy 1 lần
#df_info_comments.to_csv('models/cleaned_df_info_comments.csv')
df_info_comments = pd.read_csv('models/cleaned_df_info_comments.csv')

unusing_words = ['!','*','?','>']
#chạy mô hình cosine
cosine_sim = contentbased_filtering(df_info_comments,'cosine',unusing_words)  

#chạy mô hình gensim
# sim_model = contentbased_filtering(df_info_comments, 'gemsim', unusing_words)


                        