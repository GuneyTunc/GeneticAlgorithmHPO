Proje Teknik Spesifikasyonu: Çok Amaçlı Genetik Algoritma ile Hiperparametre Optimizasyonu (MO-GA-HPO)

1. Proje Kimliği ve Vizyonu
   -Proje Adı: MO-GA-HPO (Multi-Objective Genetic Algorithm for Hyperparameter Optimization)

-Akademik Seviye: Bilgisayar Mühendisliği 3. Sınıf - Yapay Zeka Dersi Projesi

-Temel Felsefe: "Green AI" ve "Efficiency-Aware Machine Learning". Sadece en doğru modeli değil, kaynakları en verimli kullanan modeli evrimsel yöntemlerle keşfetmek.

2. Akademik Temel ve Referans
   -Bu proje, literatürdeki en güncel ve prestijli çalışmalardan biri olan aşağıdaki makaleyi metodolojik rehber olarak kabul eder:

-Referans Makale: Morales-Hernández, A., Van Nieuwenhuyse, I., & Rojas Gonzalez, S. (2023). "A survey on multi-objective hyperparameter optimization algorithms for machine learning".

-Yayıncı: Springer - Artificial Intelligence Review.

-DOI/Link: 10.1007/s10462-022-10359-2

3. Problem Tanımı ve Çok Amaçlılık (Multi-Objective)
   -Geleneksel hiperparametre optimizasyonu (HPO) sadece hata payını minimize etmeye çalışır. Bu proje ise bir ödünleşim (trade-off) problemini çözer.
   -Hedef Fonksiyonlar:$f_1$ (Doğruluk): Sınıflandırma başarısının (Accuracy/F1-Score) maksimizasyonu. -$f_2$ (Verimlilik): Hesaplama maliyetinin (Parametre sayısı ve Inference süresi) minimizasyonu.
   -Çıktı: Tek bir çözüm yerine, bu iki hedef arasındaki dengeyi temsil eden Pareto Cephesi (Pareto Front) noktaları.

4. Sistem Mimarisi ve Genetik Algoritma (GA) Yapısı
   A. Kromozom Temsili (Genetik Kod)
   Her birey (model konfigürasyonu) aşağıdaki genlerden oluşur:

-Katman Sayısı: [1, 5] (Tamsayı)

-Nöron Sayısı: [16, 32, 64, 128, 256, 512] (Kategorik)

-Öğrenme Oranı (LR): [1e-4, 1e-1] (Sürekli/Logaritmik)

-Aktivasyon Fonksiyonu: [ReLU, Tanh, ELU] (Kategorik)

-Dropout Oranı: [0.0, 0.5] (Sürekli)
B. Evrimsel Döngü
-Başlatma: Rastgele hiperparametre setleriyle ilk popülasyonu oluştur.

-Değerlendirme (Fitness): Her bireyi belirlenen veri seti (örn. Fashion-MNIST) üzerinde kısa süreli eğit.

-Seçilim: Pareto baskınlığına göre en iyi bireyleri ebeveyn olarak seç.

-Çaprazlama (Crossover): Ebeveynlerin genlerini hibritleyerek yeni nesiller üret.

-Mutasyon: Rastgele gen değişimleri ile lokal optimumdan kaçış sağla.

5. Donanım Optimizasyonu ve Paralelleştirme
   -Proje, yerel donanım imkanlarını (Ryzen 7 7840HS ve RTX 4060) en üst seviyede kullanacak şekilde tasarlanmıştır:

-GPU Hızlandırma (RTX 4060): Model eğitimleri CUDA çekirdekleri üzerinde asenkron olarak gerçekleştirilir.

-CPU Paralelleştirme (Ryzen 7 - 16 Threads): Genetik algoritmanın popülasyon yönetimi, crossover ve mutasyon işlemleri çoklu çekirdek mimarisiyle paralel koşturulur.

-Hafıza Yönetimi: VRAM darboğazını önlemek için modeller sırayla veya küçük gruplar (batch) halinde eğitilir.

6. Proje Planı ve Kilometre Taşları (PRD)

Aşama,Süre,Hedef
Faz 1: Hazırlık,1. Hafta,Veri seti entegrasyonu ve temel PyTorch/TensorFlow model şablonunun oluşturulması.
Faz 2: GA Motoru,2. Hafta,PyGAD veya DEAP kütüphanesi ile evrimsel döngünün kodlanması.
Faz 3: Çok Amaçlılık,3. Hafta,"Fitness fonksiyonuna ""parametre sayısı"" ceza puanının eklenmesi ve Pareto mantığının kurulması."
Faz 4: Deneyler,4. Hafta,RTX 4060 üzerinde tam kapsamlı eğitim döngülerinin çalıştırılması.
Faz 5: Analiz,5. Hafta,Model A (Performans) ve Model B (Verimlilik) ayrımının yapılması ve karşılaştırmalı testler.
Faz 6: Final,6. Hafta,"Raporlama, görselleştirme (Pareto eğrisi) ve sunum hazırlığı."

7. Beklenen Çıktılar
   Performans Şampiyonu: Donanım kısıtı olmadan ulaşılabilecek en yüksek başarımlı model mimarisi.

Verimlilik Şampiyonu: Başarıdan %1-2 ödün vererek, %70-80 oranında daha az kaynak tüketen "hafif" model mimarisi.

Analitik Rapor: Evrimsel sürecin model başarısı üzerindeki etkisini gösteren veri analizleri.

Opus 4.6 (veya benzeri modeller) için Not: Bu doküman, kısıtlı donanım kaynakları altında asimetrik bir optimizasyon problemini çözmek üzere kurgulanmıştır. Analiz sırasında donanım darboğazlarını (bottlenecks) ve genetik algoritmanın yakınsama (convergence) hızını dikkate alarak strateji geliştiriniz.
