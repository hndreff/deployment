import streamlit as st
import streamlit.components.v1 as stc

from ml_app2 import run_ml_app

html_temp = """
            <div style="background-color:#21b821;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;">Food Delivery Time Estimation App </h1>
		    <h4 style="color:white;text-align:center;">Second Hexagon </h4>
		    </div>
            """

desc_temp = """
            ### Food Delivery Time Estimation App
            This app will predict your food delivery time estimation
            #### Data Source
            - https://www.kaggle.com/datasets/ranitsarkar01/porter-delivery-time-estimation/data
            """

def main():

    stc.html(html_temp)
    
    menu = ['Home', 'Machine Learning']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Welcome to Homepage")
        st.markdown(desc_temp)
    elif choice == "Machine Learning":
        # st.subheader("Welcome to Machine learning")
        run_ml_app()


if __name__ == '__main__':
    main()
