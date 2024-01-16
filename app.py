import streamlit as st
import pandas as pd
from pipline.decision_maker import Pipeline
from img_handler import ImageHandler
import os
import subprocess
import subprocess
import sklearn
import joblib

def install_package(package, version):
    try:
        subprocess.run(['pip', 'install', f'{package}=={version}'], check=True)
        print(f'Successfully installed {package} version {version}')
    except subprocess.CalledProcessError:
        print(f'Error installing {package} version {version}')

def check_and_install_dependencies():
    # Desired versions
    scikit_learn_version = '1.2.2'
    joblib_version = '1.2.0'

    # Check current versions
    try:
        current_scikit_learn_version = sklearn.__version__
        current_joblib_version = joblib.__version__
    except NameError:
        # Handle the case where the modules are not imported
        current_scikit_learn_version = 'Module not imported'
        current_joblib_version = 'Module not imported'

    print(f'Current scikit-learn version: {current_scikit_learn_version}')
    print(f'Current joblib version: {current_joblib_version}')

    # Check if versions match, and install if necessary
    if current_scikit_learn_version != scikit_learn_version:
        install_package('scikit-learn', scikit_learn_version)

    if current_joblib_version != joblib_version:
        install_package('joblib', joblib_version)

# Check and install dependencies
check_and_install_dependencies()



def main():

    st.title('Tennis Play Predictor')
    st.image('resources/Tennis.jpg')  
    st.header('Welcome to the Tennis Play Predictor!')  
    st.subheader('Introduction')
    st.write(
            """
        Welcome to the Tennis Play Predictor! This simple web app utilizes a machine learning model, specifically a decision tree, to help you decide whether you should play tennis today. The model considers various weather factors such as humidity, outlook, temperature, and windiness to make predictions.

        To get started, use the form below to input the current weather conditions. After submitting the form, the app will provide a prediction on whether it's a good day to play tennis or not.

        Enjoy!"""
    )
    # Create an instance of the Pipeline
    pipeline = Pipeline()
    
    # User Input Form
    with st.form("form"):
        st.subheader('Input the weather factors')
        st.markdown(
            """
            <style>
          
                div[data-baseweb="form"] {
                    padding: 70px;
                }
                div[data-baseweb="select"] {
                    margin-bottom: 10px;
                }
                div[data-baseweb="radio-group"] {
                    margin-bottom: 10px;
                }
                div[data-baseweb="slider"] {
                    margin-bottom: 20px;
                }
                div[data-baseweb="button"] {
                    margin-top: 20px;
                }
            </style>
            """,
            unsafe_allow_html=True)
        hot_icon = '🔥'
        mild_icon = '😊'
        cool_icon = '❄️'
        humidity_icon = '💧'
        windy_icon = '🌬️'
        sunny_icon = '🌞'
        overcast_icon = '☁️'
        rainy_icon = '🌧️'

        outlook = st.selectbox(f'Outlook in ( Sunny {sunny_icon} , Overcast {overcast_icon} , rainy {rainy_icon})', options=['Sunny', 'Overcast', 'Rainy'], key='outlook')
        temperature = st.select_slider(f' Temperature in (Mild {mild_icon}, Cool {cool_icon}, Hot {hot_icon})', options=['Mild', 'Cool', 'Hot'], key='v', value='Cool')
        humidity = st.select_slider(f'Humidity {humidity_icon}', options=['Normal', 'High'], key='hum')
        windy = st.radio(f'Windy {windy_icon}', options=[False, True])
        submitted = st.form_submit_button("Predict")

    
    if submitted:
        st.subheader('The result')
        prediction_result(pipeline, temperature, humidity, windy, outlook)




    st.write('---')
    st.subheader('Decision Tree Visualization')
    st.caption("""
    The decision tree visualization illustrates the model's decision-making process,
    depicting patterns learned from the training data.
    """)

    # Visualization Button
    tree_container = st.empty()
    if st.button("Visualize Decision Tree"):
        visualize_tree(tree_container )
    st.write('---')
    st.subheader("PlayTennis dataset")
    with st.expander("dataset used"):
        df = load_csv_data()
        st.table(df)
        if st.button("Download Dataset"):
            # Trigger the download event
            st.markdown(download_csv(df), unsafe_allow_html=True)
    image_handler = ImageHandler()
    
    





# @st.cache_data
def visualize_tree(container):
    with st.spinner("Processing..."):
        tree_fig = Pipeline.visualize_tree()
        container.pyplot(tree_fig)
        
# @st.cache_data
def getImage(prediction):
    image_handler = ImageHandler() 
    st.image(image_handler.get_image(prediction), caption=prediction, use_column_width=True)

def prediction_result(pipelines, temperature, humidity, windy, outlook):
    # Create a DataFrame for the current input data
    input_data = pd.DataFrame({
        'outlook': [outlook.lower()],
        'temp': [temperature.lower()],
        'humidity': [humidity.lower()],
        'windy': [windy]
    })

    # Make prediction
    prediction = pipelines.predict(input_data)
    
    # Display result
    if prediction == 'yes':
        st.success('Play Tennis!')
        st.toast('Play Tennis!!', icon='🎉')
        getImage(prediction)
        st.balloons()
    
    else:
        st.toast('Do not play Tennis!')
        st.info('Do not play Tennis!')
        getImage(prediction)
    
    st.write('based on your inputs')
    st.table(input_data)



import base64



# @st.cache_data
def load_csv_data():
    
    df = pd.read_csv('resources/PlayTennis.csv')
    return df

def download_csv(df, filename='PlayTennis.csv', button_text='Download Dataset...click here'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert DataFrame to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{button_text}</a>'
    return href


if __name__ == "__main__":
    main()
