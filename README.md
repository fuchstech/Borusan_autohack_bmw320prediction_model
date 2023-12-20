# BMW 3.20D Price Prediction Model
Sahibinden.com üzerinden topladığımız BMW 3.20D comfort paket ilanlarından aldığımız veriler ile hazırladığımız dataset üzerinden bir model eğittik. SVM(Support Vector Machine), Polynomal Regression, Lineer Regression, Random Forest, KNN methodları geliştirerek en iyi sonucu veren algoritmayı bulmaya çalıştık. Random Forest algoritması en yüksek doğruluğu veren algoritma oldu ve projeyi Random forest üzerinde geliştirdik. Aracın fiyatına etki eden faktörler arasında konum, değişen veya boyalı parçalar, hasar kaydı gibi verilerin etkili olduğu sonucuna vardık. Batı şehirlerinde; İstanbul, İzmir ve Edirne gibi şehirlerde araç fiyatlarının çok daha pahalı olduğu doğuya gidildikçe araç fiyatlarının düştüğü gözlemledik. 

## Installation
Programı çalıştırmak için bir anaconda dağıtımı gerekmektedir. Anaconda dağıtımını indirdikten sonra Anaconda Prompt üzerinde aşağıdaki kodları çalıştırmanız yeterli. Ayrıca dosya yollarını da değiştirmeniz gereklidir.

```
conda create -n autohack
conda activate autohack
pip install -r requirements.txt
python predict_price.py
```
