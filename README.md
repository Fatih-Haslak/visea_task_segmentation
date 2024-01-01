## Segmentation Model

**segmenation klasorünün içeriğinde (https://github.com/qubvel/segmentation_models.pytorch) github repository'si bulunmaktadır. Kurulum için dokümantasyon sayfasın linki budur-> (https://smp.readthedocs.io/en/latest/). Bu klasörde ekstra olarak benim train.py kodum , dataset klasorüm, results klasorüm , weights klasorüm, custom_utils klasorüm ve içerisindeki kodlarım, data klasorümün içerisinde ise dataloader ve color_to_label kodlarım ve config klasorüm bulunmaktadır.**

## Çalıştırma

----

**Model eğitmek için**

cd your_path/visea_task_segmentation/segmentation

python3 train.py

----

**Modeli test etmek ve sonuçları almak için**

cd your_path/visea_task_segmentation

python3 inference.py

----

## CONFIG DOSYALARI

**segmentation/config klasörünün içerisinde train ve inference kodları için .yaml uzantılı config dosyaları bulunmaktadır. Bu dosyalardan model parametreleri ve veri seti, model dosyalarının yolları belirtilebilmektedir.**

## SONUÇLAR
![](https://github.com/Fatih-Haslak/visea_task_segmentation/blob/main/segmentation/results/Figure_20.png)
![](https://github.com/Fatih-Haslak/visea_task_segmentation/blob/main/segmentation/results/Figure_1.png)
![](https://github.com/Fatih-Haslak/visea_task_segmentation/blob/main/segmentation/results/Figure_10.png)
![](https://github.com/Fatih-Haslak/visea_task_segmentation/blob/main/segmentation/results/Figure_23.png)
![](https://github.com/Fatih-Haslak/visea_task_segmentation/blob/main/segmentation/results/Figure_4.png)
![](https://github.com/Fatih-Haslak/visea_task_segmentation/blob/main/segmentation/results/Figure_11.png)
