# House-Prediction-Using-Machine-learning

![cover](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide1.PNG)

![agenda](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide2.PNG)
[Sumber data California House Prediction](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

# **CONTENTS**
##### 1. Business Problem Understanding
##### 2. Data Understanding
##### 3. Data Preprocessing
##### 4. Modeling
##### 5. Conclusion
##### 6. Recommendation

![Business](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide3.PNG)

### Context
Data set ini merupakan kumpulan data perumahan di kawasan California, Amerika Serikat yang berasal dari sensus yang dilakukan di tahun 1990 yang berisikan data-data demography (income, populasi, house occupancy) di suatu area, lokasi area (latitude, longitude) dan informasi general terkait rumah yang berada di area tersebut (number of rooms, number of bedrooms, age of the house). Jadi, meskipun mungkin tidak membantu Anda memprediksi harga perumahan saat ini seperti dataset Zillow Zestimate, ini menyediakan dataset pengantar yang dapat diakses untuk mengajari orang-orang tentang dasar-dasar Machine Learning.
### Problem Statement
Tantangan dari setiap developer perumahan adalah bagaimana menentukan harga perumahan yang tepat serta dimana lokasi untuk membangun perumahan agar tidak salah sasaran ketika menentukan harga dan lokasi perumahannya. Developer tentu saja tidak ingin membangun perumahan elit di kawasan yang notabene warga sekitarnya berpenghasilan rendah, atau membangun perumahan yang biasa saja di kawasan elit. Hal ini tentu saja akan berdampak dari penjualan properti rumah tersebut.
### Goals
Tujuan dari pemodelan ini adalah untuk menentukan harga jual dari suatu rumah berdasarkan fitur-fitur yang ada di seputaran daerah California. Adanya perbedaan pada berbagai fitur yang terdapat pada suatu properti, seperti jumlah kamar, lokasi, pendapatan rata-rata populasi dapat menambah keakuratan prediksi harga jual, yang mana dapat mendatangkan profit bagi setiap developer perumahan yang juga sesuai dengan target marketing dari developer tersebut
### Analytic Approach
Jadi, yang perlu kita lakukan adalah menganalisis data untuk dapat menemukan pola dari fitur-fitur yang ada, yang membedakan satu properti dengan yang lainnya.
### Metric Evaluation
Evaluasi metrik yang akan digunakan adalah RMSE, MAE, dan MAPE, di mana RMSE adalah nilai rataan akar kuadrat dari error, MAE adalah rataan nilai absolut dari error, sedangkan MAPE adalah rataan persentase error yang dihasilkan oleh model regresi. Semakin kecil nilai RMSE, MAE, dan MAPE yang dihasilkan, berarti model semakin akurat dalam memprediksi harga sewa sesuai dengan limitasi fitur yang digunakan.
Selain itu, kita juga bisa menggunakan nilai R-squared atau adj. R-squared jika model yang nanti terpilih sebagai final model adalah model linear. Nilai R-squared digunakan untuk mengetahui seberapa baik model dapat merepresentasikan varians keseluruhan data. Semakin mendekati 1, maka semakin fit pula modelnya terhadap data observasi. Namun, metrik ini tidak valid untuk model non-linear.

![Tawarkan](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide4.PNG)

Membuat Machine Learning untuk memprediksi harga rumah.

![ML](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide5.PNG)

Machine Learning dibutuhkan agar ketika orang membeli rumah di kawasan California, bisa diprediksi harganya sehingga error yang terjadi bisa diperkecil. 

![dataset](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide6.PNG)

longitude            --> Garis bujur, semakin tinggi semakin mejauh dari barat
latitude             --> Garis yang horizontal / mendatar semakin tinggi semakin menjauh dari utara
housing_median_age   --> Rata-rata usia rumah, semakin besar maka semakin tua
total_rooms          --> Total ruangan
total_bedrooms       --> Total kamar
population           --> Total populasi
households           --> Total kepala keluarga
median_income        --> Rata-rata pemasukan perkepala keluarga (US$)
median_houseValue    --> Rata-rata harga rumah (US$)
ocean_proximity      --> Jarak ke pantai/laut

![Modelling](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide7.PNG)

Model Algoritma yang digunakan ada , yaitu Linear Regression, KKN Regressor, DecisionTree Regressor, RandomForest Regressor, XGBoost Regressor.
Dari masing-masing algoritma akan dicari yang terbaik untuk kemudian dilakukan tuning

![RMSE](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide8.PNG)

Berdasarkan nilai RMSE, jika dilihat dari nilai rata-rata nya, XGBoost memilki nilai yang paling baik, kemudian diikuti dengan RandomForest. namun jika dilihat dari standard deviasi, KNN regressor merupakan yang paling baik, diikuti dengan Linear Regression

![MAE](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide9.PNG)

Berdasarkan nilai MAE, jika dilihat dari nilai rata-rata nya, XGBoost memilki nilai yang paling baik, kemudian diikuti dengan RandomForest. namun jika dilihat dari standard deviasi, KNN regressor merupakan yang paling baik, diikuti dengan RandomForest

![MAPE](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide10.PNG)

Berdasarkan nilai MAPE, jika dilihat dari nilai rata-rata nya, XGBoost memilki nilai yang paling baik, kemudian diikuti dengan RandomForest. Jika dilihat dari standard deviasi XGboost juga merupakan yang paling baik, diikuti dengan KNN regressor

Selanjutnya akan dilakukan prediksi pada test set dengan menggunakan 2 model benchmark terbaik yaitu XGboost dan RandomForest

![Tuned](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide11.PNG)

Melakukan prediksi pada test set dengan menggunakan model XGBoost dan hyperparameter terpilih.

![Performa](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide12.PNG)

Model mengalami peningkatan performa (nilai RMSE, MAE & MAPE berkurang) dengan dilakukannya hyperparameter tuning.
- **Sebelum Tuning** :
    - RMSE : 43951.740457
    - MAE : 29369.021115
    - MAPE : 0.167447
- **Setelah Tuning** :
    - RMSE : 43634.316318
    - MAE : 29092.170909
    - MAPE : 0.164377
    
![Perbandingan](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide12.PNG)

Berdasarkan grafik diatas, perbandingan antara nilai harga yang diprediksi dengan harga actual terlihat cukup bagus dengan membentuk suatu pola yg linear. Namun masih terdapat sedikit error yang dimana terkadang ada data yang diprediksi nilainya rendah namun nilai aktualnya tinggi. tetapi hal ini masih dalam batas yang wajar mengingat nilai MAPE yang didapat yaitu 16% dimana artinya dapat dikategorikan kedalam 'Good forecast'

![Importances](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide13.PNG)

#### Feature Importances

Untuk dapat mengetahui sebenarnya fitur apa saja yang sangat memengaruhi target (price), kita dapat mengeceknya melalui function feature importances.

![Conclusion](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide14.PNG)

# 5. Conclusion

Berdasarkan pemodelan yang sudah dilakukan, fitur 'ocean_proximity' dan 'median_income' menjadi fitur yang paling berpengaruh terhadap 'median_house_value'.
Hal ini cukup wajar artinya kita dapat mengkonfirmasi bahwa lokasi ternyata masih menjadi predictor yang paling kuat dalam menentukan harga suatu rumah. semakin rumah tersebut berada dalam area / kawasan yang elit, tentu saja harga rumah nya akan tinggi dan juga sebaliknya. Dalam kasus ini rumah yg berada di kawasan pinggir dengan view laut merupakan rumah yang paling mahal dibandingkan dengan rumah yang berada di lokasi lainnya.
Hal ini juga berbanding lurus dengan fitur median income, dimana rata-rata penghasilan seseorang dalam suatu area akan menentukan harga rumah di sekitarnya. Semakin besar rata-rata penghasilan seseorang di area tersebut, maka akan semakin mahal harga rumahnya, begitu pula sebaliknya.
Jika kita melihat berdasarkan nilai RMSE, didapati nilai RMSE cukup tinggi, hal ini dikarenakan metric RMSE memiliki beberapa kelemahan: RMSE tergantung oleh scala dari data, jadi semakin besar skala, maka nilai RMSE nya juga besar. RMSE juga dipengaruhi oleh outlier, semakin banyak outlier maka RMSE juga bisa semakin besar. Seperti yang kita ketahui data kita memiliki outlier yg cukup banyak, tapi jika outlier nya dihilangkan maka kita akan loss informasi yang banyak pula. Oleh karena itu pada kasus ini saya lebih melihat hasil pemodelan menggunakan metric MAPE yang tidak terlalu sensitive terhadap adanya outlier, dimana hasil dari metric MAPE sendiri yg sebesar 16% yang artinya persen kesalahan hasil prediksi data dibanding data actual hanya sekitar 16%. Selain itu nilai MAPE 16% artinya termasuk kedalam kategory 'Good Forecast' atau model peramalan baik.

# 6.Recommendation
Hal-hal yang dapat dilakukan untuk mengembangkan model agar lebih baik lagi :

1. Penambahan fitur-fitur yang memiliki korelasi langsung dengan harga suatu rumah, misal luas rumah, fasilitas rumah, perusahaan developernya , dll.

2. Data perlu diperbaharui karena data yang digunakan merupakan data yang sudah lama yaitu tahun 1990. data ini tentu saja sudah sangat tidak relevan dengan kondisi pada saat ini. karena adanya faktor inflasi dan sebagainya.

3. Dari sisi modeling mungkin dapat ditingkatkan dengan metode hyperparameter yang lebih baik seperti gridsearch. Metode gridsearch mencoba seluruh kombinasi hyperparameter. Sedangkan pada randomized search yang kita gunakan dalam model tidak semua kombinasi hyperparameter dicoba tetapi kita memilih secara acak dari seluruh kemungkinan kombinasi.

4. Model ini dapat digunakan untuk prediksi harga perumahan yang memiliki fitur sejenis dengan dataset California house. Karena jika dilihat dari perbandingan nilai train dan test nya, performa model cukup stabil artinya model cenderung tidak overfitting/underfitting. Namun perlu diingat kembali bahwa data ini merupakan harga rumah di tahun 1990, yang tentu saja akan jauh berbeda dengan harga rumah di tahun sekarang, ini berkaitan dengan range harga harga yang akan diprediksi, karena jika range nya melewati atau diluar range harga dalam model, maka hasilnya akan menjadi bias

![Recomendadtion](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide15.PNG)

![Terima kasih](https://github.com/dimasigit/House-Prediction-Using-Machine-learning/blob/main/Images/Slide16.PNG)
