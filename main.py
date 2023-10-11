from pathlib import Path
import streamlit as st
import os
import argparse
from PIL import Image
import pandas as pd
from yolov5.detect import run
def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('yolov5','runs', 'detect')), key=os.path.getmtime)


if __name__ == '__main__':

    st.title('Application YOLOv5 Streamlit')
    st.sidebar.image(str(Path(f'{"Static/Digiup.png"}')), width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    source = ("Image","Real Time")
    source_index = st.sidebar.selectbox("Image", range(
        len(source)), format_func=lambda x: source[x])
   
    
    conf_thres = st.sidebar.slider(
        'Confidence', 0.00, 1.00, 0.5
    )


    
    data, prediction = st.columns(2)
   
    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
    "Image", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Uploading...'):
                with data:
                    st.image(uploaded_file, caption='Image source', width=None, clamp=False, channels="RGB", output_format="auto")
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    else:
        is_valid = False
        stop=False
        
        if st.sidebar.button("Start tracking"):
            st.header('Appuyer sur Q quand vous êtes terminé')
            run(source=0,weights='Static/best13.pt')
        

          
    if is_valid:
        print('valid')
        run(source=source,weights='Static/best13.pt',conf_thres=conf_thres)

            
        with st.spinner(text='Preparing Images'):
            for img in os.listdir(get_detection_folder()):
                with prediction:
                    st.image(str(Path(f'{get_detection_folder()}') / img),caption='Prediction ', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

                
      
    st.image(str(Path(f'{"Static/Confusion_matrix.png"}')), caption='Confusion matrix', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    col1, col2 = st.columns(2)
    
with col1:
    
    st.image(str(Path(f'{"Static/F1.png"}')), caption='F1_curve', width=None, clamp=False, channels="RGB", output_format="auto")
    st.image(str(Path(f'{"Static/Precision.png"}')), caption='P_curve ', width=None, clamp=False, channels="RGB", output_format="auto")
with col2:


    st.image(str(Path(f'{"Static/Recall.png"}')), caption='R_curve ', width=None, clamp=False, channels="RGB", output_format="auto")

col3, col4 = st.columns(2)
with col3:
    st.header('Metrics')
    st.image(str(Path(f'{"Static/train.png"}')),  width=None, clamp=False, channels="RGB", output_format="auto")
    st.image(str(Path(f'{"Static/val.png"}')),  width=None, clamp=False, channels="RGB", output_format="auto")

    
with col4:
    st.header('Prédictions')

    st.image(str(Path(f'{"Static/rot.jpg"}')),  width=None, clamp=False, channels="RGB", output_format="auto")
    st.image(str(Path(f'{"Static/scab1.jpg"}')), width=None, clamp=False, channels="RGB", output_format="auto")
    st.image(str(Path(f'{"Static/blotch.jpg"}')), width=None, clamp=False, channels="RGB", output_format="auto")
    st.image(str(Path(f'{"Static/healthy.jpg"}')), width=None, clamp=False, channels="RGB", output_format="auto")