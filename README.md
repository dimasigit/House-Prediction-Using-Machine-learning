# House-Prediction-Using-Machine-learning
# EDA-Bank-Marketing-Campaign
Tim Assesment 4 :

- [Jeffry Junior Tedjasuklaksana](https://github.com/JeffreyJuinior) 
- [Hanifah Nurfahmi](https://github.com/HanifahNurfahmi) 
- [Muhammad Al-farisy](https://github.com/mhdalfarisy) 

# Business Problem
- Database Bank marketing campaigns : Data yang menggambarkan hasil pemasaran Bank Portugal. Pemasaran yang dilakukan sebagian besar didasarkan pada panggilan telepon langsung, menawarkan nasabah bank untuk menempatkan deposito. Namun banyaknya nasabah yang menolak membuka rekening deposito membuat *Tim Sales harus membuat strategi untuk menambah nasabah membuka deposito*.

# Business Goal
- Kami selaku konsultan data science akan memberikan rekomendasi strategi kepada *Tim Sales Bank Portugal* untuk menambah nasabah membuka deposito.

# Data Understanding
- Age   	: Menyimpan informasi tentang umur klien
- Job   	: Menyimpan informasi tentang bidang pekerjaan klien
- Marital 	: Menyimpan informasi tentang status pernikahan klien
- Education 	: Menyimpan informasi tentang status pendidikan klien
- Default 	: Menyimpan informasi tentang latar belakang kredit macet klien
- Campaign	: Informasi terkait jumlah kontak yang dilakukan selama campaign ini
- Housing 	: Menyimpan informasi tentang cicilan rumah klien
- Loan 	: Menyimpan informasi tentang kepemilikan hutang klien terhadap pihak lain
- Contact 	: Menyimpan informasi tentang tipe menghubungi klien melalui telepon dan cellular
- Month 	: Menyimpan informasi tentang bulan menghubungi klien
- Day_of_Week : Menyimpan informasi tentang hari terakhir menghubungi klien
- Duration 	: Menyimpan informasi tentang durasi menghubungi klien melalui telepon dan cellular (dalam hitungan detik)
- Pdays : Menyimpan informasi tentang jumlah hari yang berlalu setelah klien terakhir dihubungi dari kampanye sebelumnya
- Previous : Menyimpan informasi tentang jumlah komunikasi dengan klien yang dilakukan yang dilakukan sebelum kampanye ini dan untuk klien ini
- Poutcome: Menyimpan informasi tentang hasil dari kampanye pemasaran sebelumnya
- Emp.Var.Rate : Menyimpan informasi tentang variasi tingkat lapangan kerja regional di suatu negara
- Cons.Price.Idx : Menyimpan informasi tentang tingkat inflasi di portugal
- Cons.Conf.Id : Menyimpan informasi tentang kepercayaan klien terhadap bank
- Euriborn3m : Menyimpan informasi tentang tingkat bunga rata-rata Europa
- Nr.employeed : Menyimpan informasi tentang jumlah karyawan bank
- Y : Menyimpan informasi tentang data klien yang jadi melakukan deposit di bank

<hr>

![cover](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/1.jpg)

## *Latar Belakang : *

![LB](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/2.jpg)

## *Goal Analisa :*

![GA](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/3.jpg)

## *Penjelasan Pemilihan Dataset :*

![PMD](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/4.jpg)

## *Filter Missing Values :*

![PMV](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/5.jpg)

## *Insert New Columns :*

![INC](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/6.jpg)

## *Drop Columns :*

![DC](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/7.jpg)

### *Business Question 1 :*

#### Insight : Kebanyakan customer yang mau membuka deposito jika dihubungi 1-10 kali, berdasarkan data customer yang dihubungi 1-10 kali membuka deposito 3.725 rekening dengan total persentase 11% dan jika dihubungi lebih dari 10 kali hanya membuka deposito sebesar 4%.

![BQ1](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/8.jpg)

### *Business Question 2 :*

##### Insight : Lebih banyak orang yang mau untuk melakukan deposito saat dihubungi melalui cellular yaitu 13% dari seluruh orang yang setuju melakukan deposito. 

![BQ2](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/9.jpg)

### *Business Question 3 :*

#### Insight : Nasabah yang paling banyak untuk setuju melakukan deposit terdapat pada bulan Maret dengan total persentase 51%

![BQ3](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/10.jpg)

### *Business Question 4 :*

#### Insight : Umur mempengaruhi keputusan untuk membuka rekening deposito. Berdasarkan data di atas yang banyak membuka deposito ada pada umur diatas 59 tahun.

![BQ4](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/11.jpg)

### *Business Question 5 :*

#### Insight : Status pernikahan mempengaruhi keputusan dalam investasi deposito, dari analisa di atas ada 13% dengan status single berada di posisi paling tinggi untuk membuka rekening deposito

![BQ5](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/12.jpg)

### *Business Question 6 :*

#### Insight : Berdasarkan hasil analisa jenjang pendidikan memiliki pengaruh untuk melakukan deposito, yaitu lulusan universitas sebanyak 4%. 

![BQ6](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/13.jpg)

### *Business Question 7 :*

#### Insight : Berdasarkan hasil analisa, kesuksesan  pada campaight pertama berhasil mencapai 55%, kemudian pada campaign kedua tingkat keberhasilan naik mencapai 70%, pada campaign ketiga naik menjadi 74% kesukesannya, pada campaign ke empat naik menjadi 100% dan pada campaign trakhir turun menjadi 83%.

![BQ7](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/14.jpg)


### *Business Question 8 :*

#### Insight : Nasabah yang memiliki rumah dengan tidak memiliki hutang memiliki persentase 11% untuk membuka rekening deposito, namun nasabah yang tidak memiliki rumah dan tidak memiliki hutang memiliki persentase 10% untuk membuka deposito. Nasabah yang tidak memiliki rumah dan memiliki hutang sama persentasenya dengan nasabah yang memiliki rumah dan memiliki hutang yaitu 10%

![BQ8](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/15.jpg)

### *Saran 1-3 :*

![S13](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/16.jpg)

### *Saran 4-6 :*

![S46](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/17.jpg)

### *Saran 7-8 :*

![S78](https://github.com/mhdalfarisy/EDA-----Bank-Marketing-Campaign/blob/main/Image/18.jpg)
