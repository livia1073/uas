# Laporan Proyek Machine Learning
### Nama : Livia
### Nim : 211351075
### Kelas : Teknik Informatika Pagi B

## Domain Proyek
Cardiovascular diseases (CVDs) merupakan penyebab kematian no 1 di dunia, dengan estimasi korban berkisar 17.9 juta jiwa setiap tahun-nya, yang mana merupakan 31% penyebab kematian di seluruh dunia.
Gagal jantung merupakan hal yang di sebabkan oleh CVD dan dataset ini memiliki 12 parimeter untuk menentukan apakah anda beresiko terkena gagal jantung atau tidak.

## Business Understanding
Orang orang dengan CVD atau beresiko tinggi terdampak CVD perlu deteksi awal dan penanganan yang mana model machine learning ini bisa menjadi salah satu media yang membantu.

Bagian laporan ini mencakup:

### Problem Statements
Latar belakang permasalahan yang di hadapi:
- Perlunya deteksi awal kecenderungan gagal jantung
- Jadwal kontrol yang tidak menentu dari rumah sakit menjadi salah satu faktor gagal jantung
### Goals
Tujuan dari model machine learning ini:
- Menjadi acuan untuk deteksi awal resiko gagal jantung.
- Meningkatkan kewaspadaan anda terhadap resiko gagal jantung sehingga anda bisa cepat cepat di tangani.

    ### Solution statements
    - Model machine learning ini di dasari dari dataset yang mempelajari tingkat kematian dari gagal jantung.
    - Model machine learning ini menggunakan metode klasifikasi dengan algoritma decision tree.

## Data Understanding
Dataset yang digunakan dalam model machine learning ini adalah dataset yang disediakan di kaggle

Contoh: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- age = usia si pasien dengan tipe inputan integer
- anaemia = apakah si pasien memiliki anemia dengan tipe inputan integer namun saya rubah menjadi select box
- creatinine_phosphokinase = kadar CPK Enzyme dalam darah dengan satuan mcg/L dengan tipe inputan integer
- diabetes = apakah si pasien memiliki diabetes dengan tipe inputan integer namun dirubah menjadi select box
- ejection_fraction = persentase darah yang dikeluarkan oleh jantung setiap kali jantung memompa darah dengan satuan integer
- high_blood_pressure = Apakah pasien memiliki darah tinggi dengan tipe inputan integer namun dirubah menjadi select box
- serum_creatinine = Jumlah kandungan serum creatinine dalam darah dalam satuan mg/dL dengan tipe inputan integer
- serum_sodium = Jumlah kandungan serum sodium dalam darah dalam satuan mEq/L edngan tipe inputan integer
- sex = Jenis Kelamin pasien dengan tipe inputan integer namun dirubah menjadi select box
- smoking = Apakah pasien seorang perokok dengan tipe inputan integer namun dirubah menjadi select box
- time = Hari pasien hingga jadwal kontrol selanjutnya


## Data Preparation
Pertama tama kita siapkan dulu dataset yang akan digunakan untuk model machine learning ini. Jika sudah kita kemudian import library python yang akan di pergunakan

```bash
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import graphviz
from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler

import pickle
```

Setelah itu kita Load data yang akan dipergunakan
```bash
df = pd.read_csv('/content/heart-failure-prediction/heart_failure_clinical_records.csv')
```
Jika sudah, kita bisa cek untuk dataset nya apakah sudah terload atau tidak
```bash
df.head()
```
Selanjutnya kita cek informasi dari ssetiap kolom dataset tersebut
```bash
df.info()
```
Kita bisa visualisasikan data tersebut kedalam heatmap
```bash
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True)
```
![image](https://github.com/livia1073/uas/assets/149212033/e93089cb-c75a-4492-9d3e-9b8d360de84d)

Selanjuta kita lihat rasio jumlah kematian
```bash
sns.countplot(x='DEATH_EVENT', data=df)
plt.title('Distribution of DEATH_EVENT')
plt.xlabel('DEATH_EVENT')
plt.ylabel('Count')

plt.show()
```
![image](https://github.com/livia1073/uas/assets/149212033/48a0a094-a371-423a-84a2-8cd54e19d17c)

Lalu kita lihat pembagianya berdasarkan usia
```bash
sns.histplot(data =df, x='age',kde=True)
plt.title('Histplot')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
```
![image](https://github.com/livia1073/uas/assets/149212033/17c3dccd-fa83-4172-bbc9-4e95a86fda86)
 atau kita bias tampilkan semua kolom kedalam bentuk hist plot
 ```bash
def draw_hist(df:pd.DataFrame):
    fig, axs = plt.subplots(4, 3, figsize = (15, 12))

    for i, column in enumerate(df.columns[:-1]):
        row_index = i // 3
        col_index = i % 3
        sns.histplot(x=column, data=df, ax=axs[row_index, col_index], alpha=0.5, kde=True)
        axs[row_index, col_index].set_title(f'Hist Plot of {column}')
    plt.tight_layout()
    plt.show()

draw_hist(df)
```
![image](https://github.com/livia1073/uas/assets/149212033/0b5af544-de86-4021-81cc-f121bcf12fa1)

atau kita bisa visualkan jumlah kematian
```bash
len_live = len(df['DEATH_EVENT'][df['DEATH_EVENT'] == 0])
len_death = len(df['DEATH_EVENT'][df['DEATH_EVENT'] == 1])

arr = np.array([len_live, len_death])
labels = ['LIVING','DIED']

print(f'Total number of Living case:- {len_live}')
print(f'Total number of Death case:- {len_death}')

plt.pie(arr, labels = labels, explode=[0.2,0.0], shadow = True)
plt.show()
```
![image](https://github.com/livia1073/uas/assets/149212033/cbe5fda0-a3de-4d5f-b75c-298dbda8683e)


Lalu kita bisa visualkan jumlah kematian yang di pengaruhi oleh diabetes
```bash
patient_nhave_diabetes_0 = df['DEATH_EVENT'][df.diabetes == 0][df.DEATH_EVENT ==0]
patient_have_diabetes_1 = df['DEATH_EVENT'][df.diabetes == 1][df.DEATH_EVENT == 1]

len_d_died = len(patient_have_diabetes_1)
len_d_alive = len(patient_nhave_diabetes_0)

arr2 = np.array([len_d_died, len_d_alive])
labels = ['Died with diabetes', 'Not died with diabetes']

print(f'Total number of Died with diabetes:- {len_d_died}')
print(f'Total number of Not died with diabetes: {len_d_alive}')

plt.pie(arr2, labels=labels, explode = [0.2,0.0], shadow = True)
plt.show()
```
![image](https://github.com/livia1073/uas/assets/149212033/0ad112ff-2390-43c6-b677-9216d93c5983)

Jika sudah cukup untuk visualisasi data kita bisa melanjutkan.
Pertama kita rubah type kolom age ke integer
```bash
df["age"] = df["age"].astype(int)
```
Jika sudah kita lanjutkan ke tahap modeling

## Modeling
Pertama kita tentukan feature dan label nya
```bash
X = df.drop('DEATH_EVENT',axis=1)
y = df['DEATH_EVENT']
```
Selanjuta tentukan kondisi untuk data train dan data test nya
```bash
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=7)
```
Selanjutnya kita lakukan standarisasi data terlebih dahulu
```bash
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
Jika sudah, kita bisa melanjutkan dengan mengepush algoritmanya
```bash
dtc = DecisionTreeClassifier(
    ccp_alpha=0.0, class_weight=None, criterion='entropy',
    max_depth=4, max_features=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_samples_leaf=1,
    min_samples_split=2, min_weight_fraction_leaf=0,
    random_state=42, splitter='best'
)

model = dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

print(f"Data Train Accuracy = {accuracy_score(y_train, dtc.predict(X_train))}")
print(f"Data Test Accuracy = {dtc_acc} \n")
```
```bash
Data Train Accuracy = 0.89
Data Test Accuracy = 0.8484848484848485
```
Kita bisa lihat tingkat akurasinya, untuk data train sejumlah 89% dan untuk data test 84%


## Evaluation
Kita bisa lakukan evaluasi dengan table confusion matrix
```bash
confusion_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Oranges", xticklabels=dtc.classes_, yticklabels=dtc.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.show()
```
![image](https://github.com/livia1073/uas/assets/149212033/c28d6857-6c54-427a-aee2-7b2428264ef0)


**Rubrik/Kriteria Tambahan (Opsional)**: 
atau kita bisa visualisasikan bagaimana algoritma ini bekerja
```bash
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
plt.figure(figsize=(12,8))
tree.plot_tree(clf_en.fit(X_train, y_train))
```
![image](https://github.com/livia1073/uas/assets/149212033/67793af3-3771-490b-80a7-53068f0f451d)
  

## Deployment
[Aplikasi saya](https://decisiontreelivi.streamlit.app)
![image](https://github.com/livia1073/uas/assets/149212033/9415df6c-48ae-41b8-bb2c-62b27d978098)


**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

