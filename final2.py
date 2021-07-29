import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
import matplotlib.pyplot as plt 
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import nltk
import time
import numpy as np
top_data_df = pd.read_csv('magaza_yorumlari.csv', encoding="utf-16")
input1_top_data_df = pd.read_csv('input1.csv', encoding="utf-16")
# Stopwordleri silme işlemi (Removing the stop words)
nltk.download('stopwords')
stop_word_list = nltk.corpus.stopwords.words('turkish')
top_data_df = top_data_df[top_data_df['Görüş'].notnull()]
stop_word_list = set(stop_word_list)
top_data_df['Görüş'] = [' '.join([w for w in x.lower().split() if w not in stop_word_list]) 
    for x in top_data_df['Görüş'].tolist()]
print("Olumlu Olumsuz Sayisi:")
print(top_data_df['Durum'].value_counts())
# Duyarlılığı haritalama fonksiyonu (Function to map stars to sentiment)
def map_sentiment(Durum):
    if Durum == 'Olumlu':
        return 1
    elif Durum == 'Olumsuz':
        return 0
# Skip-gram model (sg = 1)
size = 90
window = 70
min_count = 1
workers = 12
sg = 1
# üç kategordeki duyarlılığı haritalama işlemi (Mapping stars to sentiment into three categories)
top_data_df['Durum'] = [ map_sentiment(x) for x in top_data_df['Durum']]
input1_top_data_df['Durum'] = [ map_sentiment(x) for x in input1_top_data_df['Durum']]
# Duygu dağılımını çizme (Plotting the sentiment distribution)
plt.figure()
pd.value_counts(top_data_df['Durum']).plot.bar(title="Olumlu Olumsuz Sayı Tablosu")
plt.xlabel("Durum")
plt.ylabel("Veri Sayısı")
plt.show()
# Her kategorinin ilk birkaç numarasını alma fonksiyonu (Function to retrieve top few number of each category)
top_data_df_positive = top_data_df[top_data_df['Durum'] == 1].head(3000)
top_data_df_negative = top_data_df[top_data_df['Durum'] == 0].head(3000)
top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative])
input1_top_data_df_positive = input1_top_data_df[input1_top_data_df['Durum'] == 1].head(3000)
input1_top_data_df_negative = input1_top_data_df[input1_top_data_df['Durum'] == 0].head(3000)
input1_top_data_df_small = pd.concat([input1_top_data_df_positive, input1_top_data_df_negative])
# Yeni 'tokenized_text' sütununu almak için yeni sütunu tokenize etme işlemi (Tokenize the text column to get the new column 'tokenized_text')
top_data_df_small['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in top_data_df_small['Görüş']] 
input1_top_data_df_small['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in input1_top_data_df_small['Görüş']] 
#print(top_data_df_small['tokenized_text'].head(2)) istenirse kullanılabilir ilk 2 tokenize edilen cümle
porter_stemmer = PorterStemmer()
# stemmed_tokens alma işlemi
top_data_df_small['Görüş'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in top_data_df_small['tokenized_text'] ]
top_data_df_small['Görüş'].head()
input1_top_data_df_small['Görüş'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in input1_top_data_df_small['tokenized_text'] ]
input1_top_data_df_small['Görüş'].head()
# Train Test Split fonksiyonu
def split_train_test(top_data_df_small, test_size=0.2,n_samples = 0 ):
    X_train, X_test, Y_train, Y_test = train_test_split(top_data_df_small['Görüş'], 
                                                        top_data_df_small['Durum'],                                                        
                                                        test_size=test_size 
                                                        )
    #random_state=15,shuffle_state=True, shuffle=shuffle_state,
    print("Train verisi sayısı")
    print(Y_train.value_counts())
    print("Test verisi sayısı ")
    print(Y_test.value_counts())
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    Y_train = Y_train.to_frame()
    Y_train = Y_train.reset_index()
    Y_test = Y_test.to_frame()
    Y_test = Y_test.reset_index()
    #print(X_train.head())
    return X_train, X_test, Y_train, Y_test
def findBestKnnValueAndGraf():
    randomPozitiveAndNegative()
    print("En iyi k değeri hesaplanıyor")
    i=0
    tempi = 0
    temp = 0,
    max = 0,
    knnResult = []
    knnLen = []
    for x in range(0, 51):
        knnLen.append(x)
    while i < 51:
        # Dosya yükleme (Load from the filename)
        word2vec_df = pd.read_csv(word2vec_filename)
        # Modeli başlatma (Initialize the model)
        classifier = KNeighborsClassifier()
        i=i+1
        classifier = KNeighborsClassifier(n_neighbors = i, metric='manhattan', p = 15)
        start_time = time.time()
        # Modelin uygunluğu (Fit the model)
        classifier.fit(word2vec_df, Y_train['Durum'])
        #print("Time taken to fit the model with word2vec vectors: " + str(time.time() - start_time))
        from sklearn.metrics import classification_report
        test_features_word2vec = []
        for index, row in X_test.iterrows():
            model_vector = np.mean([sg_w2v_model.wv[token] for token in row['Görüş']], axis=0).tolist()
            if type(model_vector) is list:
                test_features_word2vec.append(model_vector)
            else:
                test_features_word2vec.append(np.array([0 for i in range(90)]))
        y_pred = classifier.predict(test_features_word2vec)
        result = confusion_matrix(Y_test['Durum'], y_pred)
        #sınıflandırma raporunu yazdırdım
        result1 = classification_report(Y_test['Durum'], y_pred)
        #doğrusu skorunu yazdırdım.
        result2 = accuracy_score(Y_test['Durum'],y_pred)
        knnResult.append(result2)
        print("K=  "+str(i)+"  "+str(result2))
        if(result2>max): 
            max = result2
            tempi = i   
    i=tempi
    print("En optimal k değeri\n")  
    print(i-1)
    y = knnResult
    # karşılık gelen y axis değerleri (corresponding y axis values)
    x = knnLen  
    # noktaları çizme işlemi(plotting the points) 
    plt.plot(x, y, color='green', linestyle='dashed', linewidth = 0.5,
             marker='o', markerfacecolor='blue', markersize=1) 
    # x ve y ekseni aralığını ayarlama (setting x and y axis range)
    plt.xlim(1,50)
    plt.ylim(0.80,0.90) 
    # x eksenini adlandırmak (naming the x axis)
    plt.ylabel('Doğruluk değerleri')
    # y eksenini adlandırma (knaming the y axis)
    plt.xlabel('Komşu sayısı')
    # grafiğe başlık ekleme işlemi (giving a title to my graph)
    plt.title('En iyi k değerleri')
    # çizim işlemini gösterme fonksiyonu (function to show the plot)
    plt.grid(True, linewidth=0.5, color='#ff0000', linestyle='-')
    plt.show()
    return i-1
def randomPozitiveAndNegative():
    print("Rasgele 10 tane pozitif ve negatif cümle")
    print(top_data_df[top_data_df['Durum'] == 1].sample(10))
    print(top_data_df[top_data_df['Durum'] == 0].sample(10))
# train_test_split çağırma işlemi
X_train, X_test, Y_train, Y_test = split_train_test(top_data_df_small)

word2vec_model_file = 'word2vec_' + str(size) + '.model'
input1_word2vec_model_file = 'input1_word2vec_' + str(size) + '.model'
start_time = time.time()
stemmed_tokens = pd.Series(top_data_df_small['Görüş']).values
input1_stemmed_tokens = pd.Series(input1_top_data_df_small['Görüş']).values
# Word2Vec Modeli eğitim işlemi
w2v_model = Word2Vec(stemmed_tokens, min_count = min_count, vector_size = size, workers = workers, window = window, sg = sg)
input1_w2v_model = Word2Vec(stemmed_tokens, min_count = min_count, vector_size = 7, workers = workers, window = 5, sg = sg)
print("train word2vec'nin eğitimi için geçen süre': " + str(time.time() - start_time))
w2v_model.save(word2vec_model_file)
input1_w2v_model.save(input1_word2vec_model_file)
# Model dosyasından modeli yükleme işlemi (Load the model from the model file)
sg_w2v_model = Word2Vec.load(word2vec_model_file)
input1_sg_w2v_model = Word2Vec.load(input1_word2vec_model_file)
# Unique ID of the word
"""print("Index of the word 'urun':")
print(sg_w2v_model.wv.key_to_index["urun"])
# Toplam kelime sayısı (Total number of the words)
print(len(sg_w2v_model.wv))
# Bir kelime için word2vec vektörünün boyutunu yazdırma (Print the size of the word2vec vector for one word)
print("Length of the vector generated for a word")
print(len(sg_w2v_model.wv['urun']))
"""
# Örnek bir inceleme için vektörlerin ortalamasını alma (Get the mean for the vectors for an example review)
#print("Print the length after taking average of all word vectors in a sentence:")
#print(np.mean([sg_w2v_model.wv[token] for token in top_data_df_small['Görüş'][0]], axis=0))
# Aşağıdaki dosyada eğitim verileri için vektörleri tutma işlemi (Store the vectors for train data in following file)
word2vec_filename = 'train_review_word2vec.csv'
input1_word2vec_filename = 'input1_train_review_word2vec.csv'
with open(word2vec_filename, 'w+') as word2vec_file:
    for index, row in X_train.iterrows():
        model_vector = np.mean([sg_w2v_model.wv[token] for token in row['Görüş']], axis=0).tolist()
        
        if index == 0:
            header = ",".join(str(ele) for ele in range(90))
            word2vec_file.write(header)
            word2vec_file.write("\n")
        # Eğer çizgi varsa, sıfırların vektörü kontrolü (Check if the line exists else it is vector of zeros)
        if type(model_vector) is list:  
            line1 = ",".join( [str(vector_element) for vector_element in model_vector] )
        else:
            line1 = ",".join([str(0) for i in range(90)])
        word2vec_file.write(line1)
        word2vec_file.write('\n')
with open(input1_word2vec_filename, 'w+') as input1_word2vec_file:
    input1_model_vector = np.mean([sg_w2v_model.wv[token] for token in row['Görüş']], axis=0).tolist()
    if index == 0:
            header = ",".join(str(ele) for ele in range(90))
            word2vec_file.write(header)
            word2vec_file.write("\n")
    if type(input1_model_vector) is list:  
        line1 = ",".join( [str(vector_element) for vector_element in input1_model_vector] )
        
    else:
        line1 = ",".join([str(0) for i in range(90)])
    input1_word2vec_file.write(line1)
    input1_word2vec_file.write('\n')
#en iyi knn değerini buluyoruz
bestknnvalue = findBestKnnValueAndGraf()
#Import the KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = bestknnvalue, metric='manhattan', p = 15)
# Dosya adını yükleme işlemi (Load from the filename)
word2vec_df = pd.read_csv(word2vec_filename)
# Modeli başlatma (Initialize the model)
classifier = KNeighborsClassifier()
input1_classifier = KNeighborsClassifier()
classifier.fit(word2vec_df, Y_train['Durum'])
input1_classifier.fit(word2vec_df, Y_train['Durum'])
start_time = time.time()
# Modelin uygunluğu (Fit the model)
print("Word2vec modelinin eğitimi için geçen süre: " + str(time.time() - start_time))
test_features_word2vec = []
input1_test_features_word2vec = []
for index, row in X_test.iterrows():
    model_vector = np.mean([sg_w2v_model.wv[token] for token in row['Görüş']], axis=0).tolist()
    if type(model_vector) is list:
        test_features_word2vec.append(model_vector)
    else:
        test_features_word2vec.append(np.array([0 for i in range(90)]))
y_pred = classifier.predict(test_features_word2vec)
input1_model_vector = np.mean([input1_sg_w2v_model.wv[token] for token in row['Görüş']], axis=0).tolist()
if type(input1_model_vector) is list:
    input1_test_features_word2vec.append(model_vector)
else:
    input1_test_features_word2vec.append(np.array([0 for i in range(90)]))
input1_y_pred = input1_classifier.predict(input1_test_features_word2vec)
result = confusion_matrix(Y_test['Durum'], y_pred)
print("Karmaşıklık Matirisi:")
print(result)
#sınıflandırma raporunu yazdırdım
result1 = classification_report(Y_test['Durum'], y_pred)
print("Sınıflandırma Raporu:",)
print (result1)
#doğruluk skorunu yazdırdım.
result2 = accuracy_score(Y_test['Durum'],y_pred)
print("Doğruluk Değeri:",result2)
# x axis değerleri (x axis values)
print("Tahmin\tCümlenin Gerçek Durumu")
if(input1_y_pred ==1):
    if len(input1_top_data_df_positive) == 0:
        print("Tahmin Olumlu \nGerçek Olumsuz")
    else:
        print("Tahmin Olumlu \nGerçek Olumlu")
if(input1_y_pred ==0):
    if len(input1_top_data_df_negative) == 0:
        print("Tahmin Olumsuz \nGerçek Olumlu")
    else:
        print("Tahmin Olumsuz \nGerçek Olumsuz")

