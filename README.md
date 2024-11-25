# BMW 3.20D Price Prediction Model
<img src="https://github.com/fuchstech/Borusan_autohack_bmw320prediction_model/blob/main/images/f30.jpg" alt="BMW F30" width="200" height="270" align="right"/>

[*Borusan Autohack*](https://www.linkedin.com/posts/coderspace-io_borusan-otomotiv-autohack-tamamland%C4%B1-activity-7143257795314831360-SUYo?utm_source=share&utm_medium=member_desktop
) yarışmasında Üçüncü olan bu projede, Sahibinden.com üzerinden topladığımız BMW 3.20D comfort paket ilanlarından aldığımız veriler ile hazırladığımız dataset üzerinden bir model eğittik. SVM(Support Vector Machine), Polynomal Regression, Lineer Regression, Random Forest, KNN methodları geliştirerek en iyi sonucu veren algoritmayı bulmaya çalıştık. Random Forest algoritması en yüksek doğruluğu veren algoritma oldu ve projeyi Random forest üzerinde geliştirdik. Aracın fiyatına etki eden faktörler arasında konum, değişen veya boyalı parçalar, hasar kaydı gibi verilerin etkili olduğu sonucuna vardık. Batı şehirlerinde; İstanbul, İzmir ve Edirne gibi şehirlerde araç fiyatlarının çok daha pahalı olduğu doğuya gidildikçe araç fiyatlarının düştüğü gözlemledik. 

[Yarışma Sunumuna Buradan Ulaşabilirsiniz](https://www.canva.com/design/DAGXfUNa6Hs/KYzuO4uwgD35DdxvT8NlVw/edit?utm_content=DAGXfUNa6Hs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## Installation
Programı çalıştırmak için bir anaconda dağıtımı gerekmektedir. Anaconda dağıtımını indirdikten sonra Anaconda Prompt üzerinde aşağıdaki kodları çalıştırmanız yeterli. Ayrıca dosya yollarını da değiştirmeniz gereklidir.

1. Anaconda Prompt'u açın ve yeni bir ortam oluşturun:
   ```bash
   conda create -n autohack
   ```

2. Oluşturduğunuz ortamı etkinleştirin:
   ```bash
   conda activate autohack
   ```

3. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

4. `predict_price.py` dosyasını çalıştırın:
   ```bash
   python predict_price.py
   ```








