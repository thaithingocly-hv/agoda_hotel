import streamlit as st
import plotly.graph_objects as go

def create_container_with_color(id, color="#E4F2EC"):
    #stw(id)
    # todo: instead of color you can send in any css
    plh = st.container()
    html_code = """<div id = 'my_div_outer'></div>"""
    st.markdown(html_code, unsafe_allow_html=True)
    with plh:
        inner_html_code = """<div id = 'my_div_inner_%s'></div>""" % id
        plh.markdown(inner_html_code, unsafe_allow_html=True)
    ## applying style
    chat_plh_style = """
        <style>
            div[data-testid='stVerticalBlock']:has(div#my_div_inner_%s):not(:has(div#my_div_outer)) {
                background-color: %s;
                border-radius: 10px;
                padding: 10px;height:10px
            };
        </style>
        """
    chat_plh_style = chat_plh_style % (id, color)
    st.markdown(chat_plh_style, unsafe_allow_html=True)
    return plh

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