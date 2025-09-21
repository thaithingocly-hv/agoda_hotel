import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# from underthesea import word_tokenize, pos_tag, sent_tokenize
import warnings
from gensim import corpora, models, similarities
import re
from pyvi.ViTokenizer import tokenize
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

import streamlit as st

def clean_text(text):
    if isinstance(text, str):
        clean_text = text.lower()
        clean_text = re.sub(r'^\w\s','',clean_text)
        return clean_text
    else:
        return text

@st.cache_data  
def word_analysis(df_info, df_comments):
    df_comments['text'] = df_comments[['Score Level','Title', 'Body']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    df_comments_grouped = df_comments.groupby('Hotel ID')['text'].apply(lambda x: ' '.join(x)).reset_index()
    df_text = pd.merge(df_info, df_comments_grouped, how='left', left_on='Hotel_ID', right_on='Hotel ID')
    df_text = df_text.drop(columns=['Hotel ID'])
    df_text['Content'] = df_text[['Hotel_Description', 'text']].apply(lambda x: ' '.join(map(str, x)), axis=1)[:500]
    
    EN2VN_STOP_WORD_FILE = 'utils/english-vnmese.txt'
    VN_STOP_WORD_FILE = 'utils/vietnamese-stopwords.txt'
    EMOJI_FILE = 'utils/emojicon.txt'
    TEENCODE_FILE = 'utils/teencode.txt'
    WRONG_WORD_FILE = 'utils/wrong-word.txt'
    
    with open(EN2VN_STOP_WORD_FILE, 'r', encoding='utf-8') as file:
        stop_words = file.read()
    en2vn_stop_words = stop_words.split('\n')
    
    with open(EMOJI_FILE, 'r', encoding='utf-8') as file:
        stop_words = file.read()
    emoji_stop_words = stop_words.split('\n')
    
    with open(WRONG_WORD_FILE, 'r', encoding='utf-8') as file:
        stop_words = file.read()
    wrong_stop_words = stop_words.split('\n')
    
    with open(VN_STOP_WORD_FILE, 'r', encoding='utf-8') as file:
        stop_words = file.read()
    vn_stop_words = stop_words.split('\n')
    
    #in thường text và lọc văn bản
    df_text['Content_cleaned'] = df_text['Content'].map(clean_text)
    
    #xóa trùng
    if df_text.duplicated().sum():
        df_text = df_text.drop_duplicates()
    
    #là null 
    if df_text['Content_cleaned'].isna().sum():
        df_text = df_text.fillna('bình thường')
        
    #remove ', ""
    df_text['Content_cleaned'] = df_text.Content_cleaned.str.replace('’', '')
    df_text['Content_cleaned'] = df_text.Content_cleaned.str.replace('"', '')
    
    #convert multiple dot to 1 dot
    df_text['Content_cleaned'] = df_text.Content_cleaned.str.replace('\.+', '.', regex=True)
    
    #chuyển dữ liệu emoji_stop_words thành dictionary
    emoji_stop_word_dict = {}
    for line in emoji_stop_words:
        parts = line.split('\t')
        if len(parts) == 2:
            emoji, description = parts
            emoji_stop_word_dict[emoji] = description
    
    #xử lý các teencode
    from processor.text import TextProcessor
    text_processor = TextProcessor()

    # teencode
    df_text['Content_cleaned'] = df_text.Content_cleaned.apply(lambda x: text_processor.replace_emoji_to_text(x, emoji_dict=emoji_stop_word_dict))
    
    #remove typo text
    df_text['Content_cleaned'] = df_text.Content_cleaned.apply(lambda x: text_processor.remove_typo_tokens(x, typo_word_lst=None))
    
    ### Remove stopword
    df_text['Content_cleaned'] = df_text.Content_cleaned.apply(lambda x: text_processor.remove_stopword(x, stopwords=wrong_stop_words))
    
    ### Remove vietnamese stopword
    df_text['Content_cleaned'] = df_text.Content_cleaned.apply(lambda x: text_processor.remove_stopword(x, stopwords=vn_stop_words))
    
    #chuyển dữ liệu emoji_stop_words thành dictionary
    en2vn_stop_word_dict = {}
    for line in en2vn_stop_words:
        parts = line.split('\t')
        if len(parts) == 2:
            en, vn = parts
            en2vn_stop_word_dict[en] = vn
            
    #convert english to vietnamese
    df_text['Content_cleaned'] = df_text.Content_cleaned.apply(lambda x: text_processor.translate_english_to_vietnam(x,eng_vie_dict=en2vn_stop_word_dict))
    return df_text

@st.cache_data
def word_visual(content):
    text = "".join(content)
    stopwords = set(STOPWORDS)
    common_stopwords = ['nhân viên', 'khách sạn', 'rất', 'nơi', 'được', 'các', 'nên', 'mình', 'cũng', 'tôi', 
                        'có', 'cho', 'và', 'là', 'của', 'có thể', 'đã', 'khi', 'như', 'trong', 'tại', 'với', 'điều', 'này', 'rồi', 
                        'nhưng', 'nếu', 'thì', 'ra', 'thành phố', 'biển', 'gần', 'đi', 'ở', 'tốt', 'ăn', 'vời',
                        'sạch', 'khá', 'view','viên','nhân','lễ tân', 'vị trí','phục vụ','kỳ nghỉ','đánh giá']
    stopwords.update(common_stopwords)

    # instantiate a word cloud object
    wc = WordCloud(
        max_words=30,
        stopwords=stopwords,
        colormap='Greens'
    )

    # generate the word cloud
    wc.generate(text)
    
    # display the word clouds
    fig = plt.figure(figsize=(10, 12))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Nhận xét của khách hàng", fontsize=14)
    return fig

@st.cache_data
def contentbased_filtering(df_text, method, unusing_words):
    df_text["Content_wt"]=df_text["Content_cleaned"].apply(lambda x: tokenize(x))
    match method:
        case 'cosine':
            vectorizer = TfidfVectorizer(analyzer='word', stop_words=unusing_words)
            tfidf_matrix = vectorizer.fit_transform(df_text['Content_wt'])

            # Tính toán độ tương đồng
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            return cosine_sim
        case 'gemsim':
            pass
# ma tran: n x n  
# Hàm đề xuất khách sạn
# # với mỗi  ks, lấy nums ks tương quan nhất
@st.cache_data
def get_recommendations(df_info, hotel_id, cosine_sim, nums=5):
    idx = df_info.index[df_info['Hotel_ID'] == hotel_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums+1]
    hotel_indices = [i[0] for i in sim_scores]
    return df_info.iloc[hotel_indices]

@st.cache_data
def search_by_string(df_text,search_str,unusing_words):
    df_text["Content_wt"]=df_text["Content_cleaned"].apply(lambda x: tokenize(x))
    # Tokenize(split) the sentences into words
    content_gem = [[text for text in x.split()] for x in df_text.Content_wt]
    # Tiền xử lý dữ liệu
    content_gem_re = [[re.sub('[0-9]+','', e) for e in text] for text in content_gem] # xem xét có cần bỏ các con số hay không
    content_gem_re = [[t.lower() for t in text if not t in ['', ' ', ',', '.', '...', '-',':', ';', '?', '%', '(', ')', '+', '/', "'", '&']] for text in  content_gem_re] # kiểm tra nội dung và đưa vào các ký tự đặc biệt
    content_gem_re = [[t for t in text if not t in unusing_words] for text in content_gem_re] # stopword
    dictionary = corpora.Dictionary(content_gem_re)
    # List of features in dictionary
    dictionary.token2id    
    # Numbers of features (word) in dictionary
    feature_cnt = len(dictionary.token2id) 
    # Obtain corpus based on dictionary (dense matrix)
    corpus = [dictionary.doc2bow(text) for text in content_gem_re]
    # Use TF-IDF Model to process corpus, obtaining index
    tfidf = models.TfidfModel(corpus)
    # tính toán sự tương tự trong ma trận thưa thớt
    index = similarities.SparseMatrixSimilarity(tfidf[corpus],
                                                num_features = feature_cnt)
    search_str_wt = tokenize(search_str)
    view_content = search_str_wt.split()
    kw_vector = dictionary.doc2bow(view_content)
    sim = index[tfidf[kw_vector]]
    # sim là numpy array chứa độ tương đồng
    # Tạo DataFrame gồm 2 cột: id và sim
    df_sim = pd.DataFrame({
        "id": range(len(sim)),
        "sim": sim
    })

    # Sắp xếp theo sim giảm dần
    df_sorted_search = df_sim.sort_values(by="sim", ascending=False)
    recommend = df_sorted_search.head(6)
    return df_text.iloc[recommend.id.to_list()]
    

                    
