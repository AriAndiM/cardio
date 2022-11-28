import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import metrics

# import cv2
# import pandas as pd
# from st_aggrid import AgGrid
# import plotly.express as px
import io

with st.container():
    with st.sidebar:
        choose = option_menu("Cardio Predict", ["Home", "Deskripsi Data", "Dataset", "Preprocessing", "Modelling", "Predict"],
                             icons=['house', 'ui-checks', 'table', 'arrow-repeat', 'boxes','check2-circle'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#85BAFF"},
        }
        )

    if choose == "Home":
        st.title('Cardiovascular Disease')
        st.write('Penyakit kardiovaskular (CVD) adalah istilah bagi serangkaian gangguan yang menyerang jantung dan pembuluh darah, termasuk penyakit jantung koroner (CHD), penyakit serebrovaskular, hipertensi (tekanan darah tinggi), dan penyakit vaskular perifer (PVD). Penyebab penyakit kardiovaskular paling umum adalah aterosklerosis atau penumpukan lemak di dinding bagian dalam pembuluh darah yang mengalir ke jantung atau otak. Kondisi ini menyebabkan pembuluh darah tersumbat atau pecah.')
        logo = Image.open('jantung.png')
        st.image(logo, caption='')

    elif choose == "Deskripsi Data":
        st.title('Deskripsi Data')
        st.subheader('There are 3 types of input features:')
        st.caption("a. Objective: factual information;")
        st.caption("b. Examination: results of medical examination;")
        st.caption("c. Subjective: information given by the patient.")

        st.subheader('Features:')
        st.caption("1. Age | Objective Feature | age | int (days)")
        st.caption("2. Height | Objective Feature | height | int (cm) |")
        st.caption("3. Weight | Objective Feature | weight | float (kg) |")
        st.caption("4. Gender | Objective Feature | gender | categorical code |")
        st.caption("5. Systolic blood pressure | Examination Feature | ap_hi | int |")
        st.caption("6. Diastolic blood pressure | Examination Feature | ap_lo | int |")
        st.caption("7. Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |")
        st.caption("8. Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |")
        st.caption("9. Smoking | Subjective Feature | smoke | binary |")
        st.caption("10. Alcohol intake | Subjective Feature | alco | binary |")
        st.caption("11. Physical activity | Subjective Feature | active | binary |")
        st.caption("12. Presence or absence of cardiovascular disease | Target Variable | cardio | binary |")

    elif choose == "Dataset":
        st.subheader('Dataset Cardiovascular')
        cardio = pd.read_csv('baru-car (2).csv')
        cardio

    elif choose == "Preprocessing":
        st.title('Ini Halaman Preprocessing')

    elif choose == "Modelling":
        st.title('Model :')
        st.subheader("1. Naive Bayes")
        st.caption("Naïve Bayes Classifier merupakan sebuah metoda klasifikasi yang berakar pada teorema Bayes . Metode pengklasifikasian dg menggunakan metode probabilitas dan statistik yg dikemukakan oleh ilmuwan Inggris Thomas Bayes , yaitu memprediksi peluang di masa depan berdasarkan pengalaman di masa sebelumnya sehingga dikenal sebagai Teorema Bayes . Ciri utama dr Naïve Bayes Classifier ini adalah asumsi yg sangat kuat (naïf) akan independensi dari masing-masing kondisi / kejadian. ")
        st.caption("Rumus :")
        st.latex(r'''P(C_{k}|x) = \frac{P(C_{k})P(x|C_{k})}{P(x)}''')
        st.caption("atau bisa dituliskan :")
        st.latex(r'''posterior = \frac{prior * likelihood}{evidence}''')
        st.caption("P adalah nilai dari :")
        st.latex(r'''P(x=v|C_{k}) = (\frac{1}{\sqrt{2\pi \sigma _{k}^{2}}} exp (-\frac{(v-\mu _{k})^{2}}{2\sigma _{k}^{2}}))''')
        st.caption("Naive bayes memiliki beberapa jenis yaitu :")
        st.caption("a. Multinomial Naive Bayes")
        st.caption("sebagian besar digunakan untuk mengklasifikasi kategori dokumen. Sebuah dokumen dapat dikategorikan bertema olahraga, politik, teknologi, atau lain-lain berdasarkan frekuensi kata-kata yang muncul dalam dokumen.")
        st.caption("b. Bernouli Naive Bayes")
        st.caption("Tipe ini mirip dengan tipe Multinomial, namun klasifikasinya lebih berfokus pada hasil ya/tidak. Prediktor yang di-input adalah variabel boolean. Misalnya, prediksi atas sebuah kata muncul dalam teks atau tidak.")
        st.caption("c. Gaussian Naive Bayes")
        st.caption("Distribusi Gaussian adalah asumsi pendistribusian nilai kontinu yang terkait dengan setiap fitur berisi nilai numerik. Ketika diplot, akan muncul kurva berbentuk lonceng yang simetris tentang rata-rata nilai fitur.")

        st.subheader("2. KNN")
        st.caption("Algoritma K-Nearest Neighbor (KNN) adalah sebuah metode klasifikasi terhadap sekumpulan data berdasarkan pembelajaran data yang sudah terklasifikasikan sebelumya. Termasuk dalam supervised learning, dimana hasil query instance yang baru diklasifikasikan berdasarkan mayoritas kedekatan jarak dari kategori yang ada dalam K-NN. Tujuan dari algoritma ini adalah untuk mengklasifikasikan obyek baru berdasarkan atribut dan sample-sample dari training data.")
        st.caption("Rumus :")
        st.latex(r'''d(x,y) = \sqrt{\sum_{i=1}^{n}(x-y)^{2}}''')

        st.subheader("3. Decision Tree")
        st.caption("Decision tree adalah algoritma machine learning yang menggunakan seperangkat aturan untuk membuat keputusan dengan struktur seperti pohon yang memodelkan kemungkinan hasil, biaya sumber daya, utilitas dan kemungkinan konsekuensi atau resiko. Konsepnya adalah dengan cara menyajikan algoritma dengan pernyataan bersyarat, yang meliputi cabang untuk mewakili langkah-langkah pengambilan keputusan yang dapat mengarah pada hasil yang menguntungkan. ")
        st.caption("Rumus Entropy:")
        st.latex(r'''Entropy (S) = \sum_{i=1}^{n}-\pi * log_{2}\pi ''')
        st.caption("Rumus Gain:")
        st.latex(r'''Gain(S,A)=Entropy(S)-\sum_{i=1}^{n} * Entropy(S_{i})''')

    elif choose == "Predict":

        st.header('Parameter-Inputan')
        def input_user():
            umur = st.number_input('Umur')
            gender = st.slider('Jenis Kelamin', 1, 2, 1)
            tinggi_badan = st.number_input('Tinggi Badan')
            berat_badan = st.number_input('Berat Badan')
            sistolik = st.number_input('Tekanan Darah Sistolik')
            diastolik = st.number_input('Tekanan Darah Diastolik')
            kolestrol = st.slider('Kolestrol', 1, 3, 1)
            glukosa = st.slider('Glukosa', 1, 3, 1)
            merokok = st.slider('Merokok', 0, 1, 0)
            alkohol = st.slider('Alkohol', 0, 1, 0)
            aktivitas = st.slider('Aktivitas', 0, 1, 0)
            data = {
                'Umur': umur,
                'Jenis Kelamin': gender,
                'Tinggi Badan': tinggi_badan,
                'Berat Badan': berat_badan,
                'Tekanan_Darah_Sistolik': sistolik,
                'Tekanan_Darah_Diastolik': diastolik,
                'Kolestrol': kolestrol,
                'Glukosa': glukosa,
                'Merokok': merokok,
                'Alkohol': alkohol,
                'Aktivitas': aktivitas
            }

            fitur = pd.DataFrame(data, index=[0])
            return fitur

        #inputan
        st.subheader('Load Data Input')
        df = input_user()
        st.write(df)

        #dataset
        cardio = pd.read_csv('baru-car (2).csv')

        #data y_training
        y = cardio['cardio'].values

        st.subheader('Load Data Cardio Terbaru')
        x = cardio.drop(columns=['id','cardio'])
        x

        #Normalisasi
        st.subheader('Normalisasi Data')
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(x)
        features_names = x.columns.copy()
        scaled_features = pd.DataFrame(scaled, columns = features_names)
        scaled_features

        import joblib
        file_name_norm = "norm.sav"
        joblib.dump(scaler, file_name_norm)

        #Model Gaussian 
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        training, test = train_test_split(scaled, train_size = 0.8, test_size = 0.2, shuffle = False)
        training_label, test_label = train_test_split(y, train_size = 0.8, test_size = 0.2, shuffle = False)

        gnb = GaussianNB()
        gnb.fit(training, training_label)
        prediksi = gnb.predict(df)
        prediksi_probas = gnb.predict_proba(df)
        prediksi[0]

        # st.subheader('Skore :', gnb.accuracy_score(test, test_label))

        # st.subheader('Class Label')
        # target_names = cardio.cardio.unique()
        # st.write(target_names)

        # st.subheader('Prediksi')
        # st.write(target_names[prediksi])

        # #save model gaussian
        # file_name_gnb = "Model Gaussian.sav"
        # joblib.dump(gnb, file_name_gnb)


        # st.subheader('Akurasi Gaussian')
        # # prediksi = gnb.predict(test)


        # #st.write('df : ', prediksi)
        # #prediksi_probas = gnb.predict_proba(df)
        # accuracy = metrics.accuracy_score(test_label, y_pred)*100
        # st.write(accuracy)

        # st.subheader('Prediksi (Hasil Klasifikasi)')
        # hasil = target_names[prediksi]
        # hasil





