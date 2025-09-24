
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def gauge_chart(head, x,bar_color):
    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = x,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': head, 'font': {'size': 11}},
    gauge = {
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': bar_color},
        'bar': {'color': bar_color},
        'bgcolor': "white",
        'steps': [
            {'range': [0, 7], 'color': 'lightgray'},
            {'range': [7, 9], 'color': 'gray'}],
        'threshold' : {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': 9.8}
        }))
    fig.update_layout(
        height=165,
        margin=dict(l=0, r=0, t=0, b=0, pad=0), # Left, Right, Top, Bottom margins in pixels
    )
    return fig

def clean_text(text):
    if isinstance(text, str):
        clean_text = text.lower()
        clean_text = re.sub(r'^\w\s','',clean_text)
        return clean_text
    else:
        return text

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

# ma tran: n x n  
# Hàm đề xuất khách sạn
# # với mỗi  ks, lấy nums ks tương quan nhất
def get_recommendations(df_info,hotel_id, nums=5):
    import pickle
    with open('models/cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
    print(f'Vào get_recommendations: {cosine_sim}')
    idx = df_info.index[df_info['Hotel_ID'] == hotel_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums+1]
    hotel_indices = [i[0] for i in sim_scores]
    return df_info.iloc[hotel_indices]

def search_by_string(df, search_str):
    df_sim_model = pd.read_csv("models/gensim.csv")
    dictionary = pd.read("models/gensim.dict")
    search_vector = dictionary.doc2bow(search_str.lower().split())
    from gensim.models import TfidfModel
    model = TfidfModel()
    tfidf = model.load("models/tfidf_model")
    sim = df_sim_model[tfidf[search_vector]]
    # sim là numpy array chứa độ tương đồng
    # Tạo DataFrame gồm 2 cột: id và sim
    df_sim = pd.DataFrame({
        "id": range(len(sim)),
        "sim": sim
    })
    # Sắp xếp theo sim giảm dần
    df_sorted_search = df_sim.sort_values(by="sim", ascending=False)
    recommend = df_sorted_search.head(6)
    return df.iloc[recommend.id.to_list()]