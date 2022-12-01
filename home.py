import streamlit as st
import streamlit.components.v1 as html
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from streamlit_option_menu import option_menu
from PIL import Image
import io

with st.container():
    with st.sidebar:
        choose = option_menu("Cardio Predict", ["Home", "Deskripsi Data", "Dataset", "Preprocessing", "Modelling", "Predict"],
                             icons=['house', 'ui-checks', 'table', 'arrow-repeat', 'boxes','check2-circle'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "10A19D"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#3D5656"},
        }
        )

    if choose == "Home":
        st.markdown('<h1 style = "text-align: center;"> Cardiovascular Disease </h1>', unsafe_allow_html = True)
        st.markdown('<div style ="text-align: justify;"> Penyakit kardiovaskular (CVD) adalah istilah bagi serangkaian gangguan yang menyerang jantung dan pembuluh darah, termasuk penyakit jantung koroner (CHD), penyakit serebrovaskular, hipertensi (tekanan darah tinggi), dan penyakit vaskular perifer (PVD). Penyebab penyakit kardiovaskular paling umum adalah aterosklerosis atau penumpukan lemak di dinding bagian dalam pembuluh darah yang mengalir ke jantung atau otak. Kondisi ini menyebabkan pembuluh darah tersumbat atau pecah. </div>', unsafe_allow_html = True)
        logo = Image.open('jantung.png')
        st.image(logo, caption='')

    elif choose == "Deskripsi Data":
        # st.markdown('<h1 style = "text-align: center;"> Deskripsi Data </h1>', unsafe_allow_html = True)
        # st.markdown('<h2>Ada 3 jenis fitur input :</h2>', unsafe_allow_html = True)
        # st.markdown('<ol type = "a"><li>Objektif : factual information;</li><li>Penelitian / penyelidikan : hasil pemeriksaan medis;</li><li>Subjektif : informasi yang diberikan oleh pasien.</li></ol>', unsafe_allow_html = True)
        st.markdown('<h1 style = "text-align: center;"> Fitur Dataset: </h1><ol type = "1" style = "text-align: justify; background-color: #f2a916; padding: 30px; border-radius: 20px;"><li>Age merupakan umur dari pasien yang diukur dalam satuan hari dengan tipe data int.</li><li>Height merupakan tinggi badan dari pasien dalam satuan cm yang diukur menggunakan alat stature meter dengan tipe data integer.</li><li>Weight merupakan berat badan dari pasien dalam satuan kg yang diukur menggunakan alat timbangan injak dengan tipe data float.</li><li>Gender merupakan jenis kelamin dari pasien. Jenis kelamin bertipe biner yaitu perempuan dan laki-laki.</li><li>Systolic blood pressure (tekanan darah sistolik) merupakan tekanan ketika jantung pasien memompa darah ke seluruh tubuh .Tekanan darah sistolik dapat diukur menggunakan alat tensimeter.</li><li>Diastolic blood pressure (tekanan darah diastolik) merupakan tekanan ketika darah masuk ke dalam jantung. Tekanan darah diastolik dapat diukur menggunakan alat tensimeter.</li><li>Cholesterol (kolesterol) merupakan lemak mirip zat lilin yang terdapat dalam darah pasien. Kolesterol dapat diketahui dengan cara melakukan pemeriksaan ke rumah sakit atau puskesmas terdekat Terdapat 3 tingkatan kolesterol yaitu 1: normal yaitu kadar kolesterol < 200 (mg/dL), 2: di atas normal yaitu kadar kolesterol antara 200 s/d 239 (mg/dL), 3: jauh di atas normal yaitu kadar kolesterol > 240 (mg/dL).</li><li>Glucose (glukosa) merupakan senyawa organik dalam bentuk karbohidrat berjenis monosakarida. Kadar glukosa dapat diukur di laboratorium atau dengan alat glukometer. 1 : Glukosa normal kadarnya sekitar 100 - 160 mg/dL, 2 : Glukosa diatas normal kadarnya sekitar 160 - 240 mg/dL, 3 : Glukosa jauh diatas normal kadarnya di atas 240 mg/dL.</li><li>Smoking merupakan kondisi pasien sedang merokok atau tidak</li><li>Alcohol intake merupakan kondisi pasien apakah mengkonsumsi alkohol atau tidak.</li><li>Physical activity merupakan kondisi pasien apakah aktif berolahraga atau tidak.</li><li>Presence or absence of cardiovascular disease merupakan hasil diagnosa apakah pasien mengidap penyakit cardiovascular.</li></ol>', unsafe_allow_html = True)

    elif choose == "Dataset":
        st.header('Dataset Cardiovascular')
        cardio = pd.read_csv('cardiovascular.csv')
        cardio

    elif choose == "Preprocessing":
        st.title("Preprocessing")
        st.caption("**_Preprocessing_** adalah sebuah pengolahan data mentah sebelum data tersebut di proses.")
        st.subheader("Normalisasi")
        st.caption("**_Normalisasi_** adalah proses untuk melakukan transformasi dari format data asli menjadi format yang lebih efisien. Contohnya seperti mengubah data asli menjadi data yang bernilai antara 0 - 1. Berikut normalisasi menggunakan MinMax. Rumus MinMax :")
        st.latex(r'''x^{'} = \frac{x - x_{min}}{x_{max}-x_{min}}''')

        st.subheader("Data tanpa label / class")
        #dataset
        cardio = pd.read_csv('cardiovascular.csv')
        #data y_training
        y = cardio['cardio'].values
        x = cardio.drop(columns=['id','cardio'])
        x

        #Normalisasi
        st.subheader('Normalisasi Data Menggunakan MinMax')

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(x)
        features_names = x.columns.copy()
        scaled_features = pd.DataFrame(scaled, columns = features_names)
        scaled_features
        
    elif choose == "Modelling":
        #dataset
        cardio = pd.read_csv('cardiovascular.csv')

        #data y_training
        y = cardio['cardio'].values
        #drop fitur id dan cardio
        x = cardio.drop(columns=['id','cardio'])
        
        #Normalisasi
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(x)
        features_names = x.columns.copy()
        scaled_features = pd.DataFrame(scaled, columns = features_names)
                
        #Model Gaussian 
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        #Model Knn
        from sklearn.neighbors import KNeighborsClassifier
        #Model Decisiom Tree
        from sklearn import tree
        from sklearn.tree import DecisionTreeClassifier
        
        #Splitting Data
        training, test = train_test_split(scaled, train_size = 0.8, test_size = 0.2, shuffle = False)
        training_label, test_label = train_test_split(y, train_size = 0.8, test_size = 0.2, shuffle = False)
        
        st.header('Modelling')
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
        
        gnb = GaussianNB()
        gnb.fit(training, training_label)
        acc_gnb = gnb.score(test, test_label)
        st.write('**_Akurasi Gaussian :_**', acc_gnb*100, '**_%_**')
        
        st.subheader("2. K-Nearest Neighbors (K-NN)")
        st.caption("Algoritma K-Nearest Neighbor (KNN) adalah sebuah metode klasifikasi terhadap sekumpulan data berdasarkan pembelajaran data yang sudah terklasifikasikan sebelumya. Termasuk dalam supervised learning, dimana hasil query instance yang baru diklasifikasikan berdasarkan mayoritas kedekatan jarak dari kategori yang ada dalam K-NN. Tujuan dari algoritma ini adalah untuk mengklasifikasikan obyek baru berdasarkan atribut dan sample-sample dari training data.")
        st.caption("Rumus :")
        st.latex(r'''d(x,y) = \sqrt{\sum_{i=1}^{n}(x-y)^{2}}''')
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(training, training_label)
        acc_knn = knn.score(test, test_label)
        st.write('**_Akurasi KNN :_**', acc_knn*100, '**_%_**')

        st.subheader("3. Decision Tree")
        st.caption("Decision tree adalah algoritma machine learning yang menggunakan seperangkat aturan untuk membuat keputusan dengan struktur seperti pohon yang memodelkan kemungkinan hasil, biaya sumber daya, utilitas dan kemungkinan konsekuensi atau resiko. Konsepnya adalah dengan cara menyajikan algoritma dengan pernyataan bersyarat, yang meliputi cabang untuk mewakili langkah-langkah pengambilan keputusan yang dapat mengarah pada hasil yang menguntungkan. ")
        st.caption("Rumus Entropy:")
        st.latex(r'''Entropy (S) = \sum_{i=1}^{n}-\pi * log_{2}\pi ''')
        st.caption("Rumus Gain:")
        st.latex(r'''Gain(S,A)=Entropy(S)-\sum_{i=1}^{n} * Entropy(S_{i})''')
        
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        acc_dt = dt.score(test, test_label)
        st.write('**_Akurasi Decision Tree :_**', acc_dt*100, '**_%_**')

    elif choose == "Predict":
        # form data kesehatan
        st.markdown('<h1 style = "text-align: center;"> Prediksi Cardiovascular Diseases </h1><h3 style = "text-align: center; color: orange;"> Masukkan Data Kesehatan Anda </h3>', unsafe_allow_html = True)
        #dataset
        cardio = pd.read_csv('cardiovascular2.csv')

        # data y_training
        y = cardio['cardio'].values
        # data terbaru
        x = cardio.drop(columns=['id','cardio'])

        #Normalisasi
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(x)
        features_names = x.columns.copy()
        scaled_features = pd.DataFrame(scaled, columns = features_names)

        #Model Gaussian 
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import tree
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        #splitting data
        X_train, X_test = train_test_split(scaled, train_size = 0.8, test_size = 0.2, shuffle = False)
        y_train, y_test = train_test_split(y, train_size = 0.8, test_size = 0.2, shuffle = False)

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

        inputan = [umur, gender, tinggi_badan, berat_badan, sistolik, diastolik, kolestrol, glukosa, merokok, alkohol, aktivitas]
        x_min = x.min()
        x_max = x.max()
        norm_input = ((inputan - x_min)/(x_max - x_min))
        norm_input = np.array(norm_input).reshape(1, -1)

        pilih_model = st.radio(
            "Pilih Model",
            ('None','Gausian Naive Bayes', 'K-Nearest Neighbors (K-NN)', 'Decision Tree'))

        if pilih_model == 'Gausian Naive Bayes':
            cek = st.button("Cek Diagnosa")
            if cek:
                gnb = GaussianNB()
                gnb.fit(X_train, y_train)
                prediksi = gnb.predict(X_test)
                pred = gnb.predict(norm_input)
                if(pred == 0):
                    st.markdown('Dengan diagnosa model **_Gaussian Naive Bayes_**', unsafe_allow_html = True)
                    st.write('Akurasi : ',round(gnb.score(X_test, y_test), 2), '%')
                    st.markdown('Anda dinyatakan **_negatif Cardiovascular_**', unsafe_allow_html=True)
                elif(pred == 1):
                    st.markdown('Dengan diagnosa model **_Gaussian Naive Bayes_**', unsafe_allow_html = True)
                    st.write('Akurasi : ',round(gnb.score(X_test, y_test), 2), '%')
                    st.caption('Anda dinyatakan **_positif_** Cardiovascular')

        elif option == 'K-Nearest Neighbors (K-NN)':
            cek = st.button("Cek Diagnosa")
            if cek:
                knn = KNeighborsClassifier()
                knn.fit(X_train, y_train)
                prediksi = knn.predict(X_test)
                pred = knn.predict(norm_input)
                if(pred == 0):
                    st.markdown('Dengan diagnosa model **_K-Nearest Neighhbors_**', unsafe_allow_html = True)
                    st.write('Akurasi : ',round(knn.score(X_test, y_test), 2), '%')
                    st.markdown('Anda dinyatakan **_negatif Cardiovascular_**', unsafe_allow_html=True)
                elif(pred == 1):
                    st.markdown('Dengan diagnosa model **_K-Nearest Neighhbors_**', unsafe_allow_html = True)
                    st.write('Akurasi : ',round(knn.score(X_test, y_test), 2), '%')
                    st.caption('Anda dinyatakan **_positif_** Cardiovascular')

        elif option == 'Decision Tree':
            cek = st.button("Cek Diagnosa")
            if cek:
                dt = DecisionTreeClassifier()
                dt.fit(X_train, y_train)
                prediksi = dt.predict(X_test)
                pred = dt.predict(norm_input)
                if(pred == 0):
                    st.markdown('Dengan diagnosa model **_Decision Tree_**', unsafe_allow_html = True)
                    st.write('Akurasi : ',round(dt.score(X_test, y_test), 2), '%')
                    st.markdown('Anda dinyatakan **_negatif Cardiovascular_**', unsafe_allow_html=True)
                elif(pred == 1):
                    st.markdown('Dengan diagnosa model **_K-Nearest Neighhbors_**', unsafe_allow_html = True)
                    st.write('Akurasi : ',round(dt.score(X_test, y_test), 2), '%')
                    st.caption('Anda dinyatakan **_positif_** Cardiovascular')

        elif option == 'None':
            st.write('Pilih Model')
            



#----------------------------------------------------------------
#indentasi
        # gnb = GaussianNB()
        # filename = "GaussianNB.pkl"

        # gnb.fit(X_train, y_train)
        # prediksi = gnb.predict(X_test)
        # inputan = [umur, gender, tinggi_badan, berat_badan, sistolik, diastolik, kolestrol, glukosa, merokok, alkohol, aktivitas]
        # x_min = x.min()
        # x_max = x.max()
        # norm_input = ((inputan - x_min)/(x_max - x_min))
        # norm_input = np.array(norm_input).reshape(1, -1)

        # pred = gnb.predict(norm_input)

        # st.write('score :', gnb.score(X_test, y_test))
        # cek = st.button("Cek Diagnosa")

        # if cek:
        #     pred[0]
        #     if(pred == 0):
        #         st.caption('negatif')
        #     elif(pred == 1):
        #         st.caption('positif')
#----------------------------------------------------------------
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





