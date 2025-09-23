import streamlit as st
from streamlit_option_menu import option_menu
#import toml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import func

#DATA
file_info = 'models/cleaned_hotel_info.csv'
file_comments = 'data/hotel_comments.csv'

#PROCESSING
df_comments = pd.read_csv(file_comments)
df_info = pd.read_csv(file_info)

#GUI
st.set_page_config(page_title="Recommendation system", page_icon="img/logo.png", layout="wide")

col1, col2 = st.columns([0.2, 0.7])
with col1:
    st.image("img/agoda.svg", width=80)
with col2:
    st.markdown("Máy bay + khách sạn | Chỗ ở | Phương tiện di chuyển")
choice= option_menu(None,
        options=["Trang chủ", "Về dự án"], 
        icons=["house", "people"], 
        default_index=0,
        orientation = "horizontal",
        styles={
            "container": {"background-color": "#fafafa"},
            "icon": {"color": "orange"}, 
            "nav-link": {"text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#8aa7ef", "font-family": "Roboto", "color": "white"},
        }  
    )
match choice:
    case "Về dự án":
        st.subheader("Đồ án tốt nghiệp khóa học Khoa học dữ liệu.")
        st.markdown("""Nơi học tập: Trung tâm tin học ĐH Khoa học Tự nhiên""", unsafe_allow_html=True)
    case "Trang chủ":
        expander = st.expander("Tìm kiếm")
        with expander:
            col1, col2 = st.columns(2)
            with col1:
                w_search = st.radio("Tiếp cận bằng cách:", ["Khách sạn","Hạng sao", "Từ tìm kiếm"])
            with col2:
                if w_search == "Khách sạn":
                    #ks = ("Email", "Home phone", "Mobile phone")
                    selected_company_info = st.selectbox(":airplane: Khách sạn?", df_info["Hotel_Name"].sort_values()[::100])
                elif w_search == "Hạng sao":
                    stars = st.select_slider(":yellow[:material/hotel_class:]", options=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
                    st.write("Đã chọn hạng", stars)
                else:
                    #gợi ý một số từ khóa
                    options = ["Khách sạn mới", "phòng ngủ rộng", "gần biển", "có hồ bơi"]
                    selection = st.pills(":material/touch_triple: Từ tìm kiếm phổ biến", options, selection_mode="multi")
                    st.markdown(f"Your selected options: {selection}.")

                    #tùy ý chọn từ khóa
                    user_search = st.text_input(":material/more: Bạn cần?","...")
                submit = st.button("Tìm kiếm", icon=":material/search:")
            #expander.button("Reset", type="primary")
                    
        if submit:    
            st.markdown("**Kết quả tìm kiếm**")
            tab1, tab2, tab3, tab4 = st.tabs([":eye: Thông tin", ":gem: Insights",":stars: Đánh giá", ":link: Gợi ý khách sạn tương tự"],)
            match w_search:
                case "Khách sạn":
                    df = df_info[df_info["Hotel_Name"] == selected_company_info]
                    hotel_id = df['Hotel_ID'].values[0]
                    with tab1:
                        st.subheader(f":rainbow[{selected_company_info}]")
                        st.write(f"**Hạng:** {df['Hotel_Rank'].values[0].split()[0]} :star:")
                        st.write(f"**Địa chỉ:** {df['Hotel_Address'].values[0]}")
                        st.write(f"**Mô tả:** {df['Hotel_Description'].values[0][:600]}")
                    with tab2:
                        st.subheader(f":rainbow[{selected_company_info}]")
                        col1,col2 = st.columns([0.2,0.8])
                        with col1:
                            fig = func.gauge_chart('Vị trí',df['Location'].values[0], 'blue')
                            st.plotly_chart(fig)
                            fig = func.gauge_chart('Độ sạch',df['Cleanliness'].values[0], 'orange')
                            st.plotly_chart(fig)
                            fig = func.gauge_chart('Dịch vụ',df['Service'].values[0], 'pink')
                            st.plotly_chart(fig)
                            fig = func.gauge_chart('Tiện ích',df['Facilities'].values[0], 'green')
                            st.plotly_chart(fig)
                            fig = func.gauge_chart('Đáng giá',df['Value_for_money'].values[0], 'yellow')
                            st.plotly_chart(fig)
                            fig = func.gauge_chart('Chất lượng',df['Comfort_and_room_quality'].values[0], 'purple')
                            st.plotly_chart(fig)
                        with col2:
                            st.markdown("<div style='text-align: center'>Quốc tịch</div>", unsafe_allow_html=True)
                            df_hotel_comments = df_comments[df_comments["Hotel ID"]== hotel_id]
                            fig = px.funnel(df_hotel_comments['Nationality'].value_counts())
                            st.plotly_chart(fig)
                    with tab3:
                        st.subheader(f":rainbow[{selected_company_info}]")
                        x = f"Số lượng người đánh giá: {df['comments_count'].values[0]}"
                        st.badge(x, color='primary', width=200)
                        
                        #get hình wordcloud
                        df_text = func.word_analysis(df, df_hotel_comments)
                        fig = func.word_visual(df_text.Content_cleaned)
                        st.pyplot(fig)
                    with tab4:
                        st.write("Các khách sạn tương tự:")
                        st.subheader(f":rainbow[{selected_company_info}]")
                        df_recommendations = func.get_recommendations(df_info,hotel_id, 6)
                        #st.dataframe(df_recommendations)
                        row1 = st.columns(3)
                        row2 = st.columns(3)
                        i = 0
                        for col in row1 + row2:
                            with col.expander(df_recommendations['Hotel_Name'].iloc[i][:25]):
                                st.write(f"Hạng: {df_recommendations['Hotel_Rank'].iloc[i]} :star:")
                            i+=1

                case "Hạng sao":
                    pass
                case "Từ tìm kiếm":
                    # Trường hợp khách hàng nhập thông tin tìm kếm.
                    st.write(f'từ được chọn từ ví dụ: {selection}')
                    st.write(f'từ khóa được nhập: {user_search}')
                    search_str = selection.append(user_search)
                    st.write("Từ khóa cần tìm",search_str)
                    # HV cần xử lý chi tiết phần này
                    # Ở đây xem như search_str đã được tiền xử lý
                    df = func.search_by_string(df_info,search_str)
                    st.dataframe(df)
                    
                    
            
            
    
    
st.markdown('<div style="text-align: center;"><div>----------</div><p>GVHD: Khuất Thùy Phương<br/>HV: Thái Thị Ngọc Lý<br/>09.2025</p></div>',unsafe_allow_html=True)
