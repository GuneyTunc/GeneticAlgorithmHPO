# Sampiyon Modeller - Basari ve Hata Metrikleri
Fitness epoch: 5 | Optimizer: Adam | Batch: 512

## Performance Champion

Mimari: 2 katman, 512 noron, relu, lr=0.000939, dropout=0.105

### Validation seti

| Metrik | Deger |
|--------|-------|
| Accuracy | 0.8746 (87.46%) |
| F1 (macro) | 0.8743 |
| Hata orani (1-accuracy) | 0.1254 (12.54%) |
| Hamming loss | 0.1254 |
| Log loss | 0.3415 |
| Zero-one loss | 0.1254 |
| Brier score (macro) | 0.0177 |
| Macro FPR | 0.0139 |
| Macro FNR | 0.1237 |
| Yanlis siniflandirma | 1129/9000 |

**Sinif bazli hata orani (1 - recall):**

| Sinif | Hata orani | Yanlis | Destek |
|-------|------------|--------|--------|
| T-shirt/top | 20.0% | 194 | 970 |
| Trouser | 1.9% | 18 | 927 |
| Pullover | 24.4% | 217 | 890 |
| Dress | 7.1% | 61 | 858 |
| Coat | 12.7% | 110 | 869 |
| Sandal | 4.4% | 39 | 892 |
| Shirt | 36.1% | 340 | 942 |
| Sneaker | 8.4% | 70 | 832 |
| Bag | 3.3% | 29 | 885 |
| Ankle boot | 5.5% | 51 | 935 |

### Test seti

| Metrik | Deger |
|--------|-------|
| Accuracy | 0.8732 (87.32%) |
| F1 (macro) | 0.8723 |
| Hata orani (1-accuracy) | 0.1268 (12.68%) |
| Hamming loss | 0.1268 |
| Log loss | 0.3556 |
| Zero-one loss | 0.1268 |
| Brier score (macro) | 0.0185 |
| Macro FPR | 0.0141 |
| Macro FNR | 0.1268 |
| Yanlis siniflandirma | 1268/10000 |

**Sinif bazli hata orani (1 - recall):**

| Sinif | Hata orani | Yanlis | Destek |
|-------|------------|--------|--------|
| T-shirt/top | 20.2% | 202 | 1000 |
| Trouser | 3.0% | 30 | 1000 |
| Pullover | 24.0% | 240 | 1000 |
| Dress | 8.4% | 84 | 1000 |
| Coat | 14.7% | 147 | 1000 |
| Sandal | 4.4% | 44 | 1000 |
| Shirt | 35.5% | 355 | 1000 |
| Sneaker | 8.7% | 87 | 1000 |
| Bag | 2.7% | 27 | 1000 |
| Ankle boot | 5.2% | 52 | 1000 |

## Efficiency Champion

Mimari: 1 katman, 16 noron, elu, lr=0.002247, dropout=0.000

### Validation seti

| Metrik | Deger |
|--------|-------|
| Accuracy | 0.8554 (85.54%) |
| F1 (macro) | 0.8553 |
| Hata orani (1-accuracy) | 0.1446 (14.46%) |
| Hamming loss | 0.1446 |
| Log loss | 0.4196 |
| Zero-one loss | 0.1446 |
| Brier score (macro) | 0.0211 |
| Macro FPR | 0.0161 |
| Macro FNR | 0.1430 |
| Yanlis siniflandirma | 1301/9000 |

**Sinif bazli hata orani (1 - recall):**

| Sinif | Hata orani | Yanlis | Destek |
|-------|------------|--------|--------|
| T-shirt/top | 17.8% | 173 | 970 |
| Trouser | 2.9% | 27 | 927 |
| Pullover | 21.6% | 192 | 890 |
| Dress | 11.2% | 96 | 858 |
| Coat | 19.1% | 166 | 869 |
| Sandal | 8.3% | 74 | 892 |
| Shirt | 41.2% | 388 | 942 |
| Sneaker | 7.5% | 62 | 832 |
| Bag | 6.0% | 53 | 885 |
| Ankle boot | 7.5% | 70 | 935 |

### Test seti

| Metrik | Deger |
|--------|-------|
| Accuracy | 0.8450 (84.50%) |
| F1 (macro) | 0.8443 |
| Hata orani (1-accuracy) | 0.1550 (15.50%) |
| Hamming loss | 0.1550 |
| Log loss | 0.4334 |
| Zero-one loss | 0.1550 |
| Brier score (macro) | 0.0221 |
| Macro FPR | 0.0172 |
| Macro FNR | 0.1550 |
| Yanlis siniflandirma | 1550/10000 |

**Sinif bazli hata orani (1 - recall):**

| Sinif | Hata orani | Yanlis | Destek |
|-------|------------|--------|--------|
| T-shirt/top | 19.2% | 192 | 1000 |
| Trouser | 4.4% | 44 | 1000 |
| Pullover | 23.5% | 235 | 1000 |
| Dress | 12.9% | 129 | 1000 |
| Coat | 23.9% | 239 | 1000 |
| Sandal | 8.4% | 84 | 1000 |
| Shirt | 42.2% | 422 | 1000 |
| Sneaker | 7.1% | 71 | 1000 |
| Bag | 6.4% | 64 | 1000 |
| Ankle boot | 7.0% | 70 | 1000 |

