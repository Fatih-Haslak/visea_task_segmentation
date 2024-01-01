## Segmentation Model

**segmenation klasorünün içeriğinde (https://github.com/qubvel/segmentation_models.pytorch) github repository'si bulunmaktadır. Kurulum için dokümantasyon sayfasın linki budur-> (https://smp.readthedocs.io/en/latest/). Bu klasörde ekstra olarak benim datalarım ve diğer kodlarım bulunmaktadır**

## Çalıştırma

**Model eğitmek için**

cd your_path/segmentation

python3 train.py

**Modeli test etmek ve sonuçları almak için**

cd your_path

python3 inference.py

**Config**

segmentation/config dosyasının içerisinde train ve inference kodları için config dosyaları bulunmaktadır. Bu dosyalardan model parametreleri ve veri seti, model dosyalarının yolları belirtilebilmektedir.
