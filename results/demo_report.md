# Rapport d'Évaluation - Restauration Vidéo

**Date d'évaluation:** 2025-05-29T20:15:40.849446

**Vidéo originale:** data/input/demo_color.mp4

## Résumé des Résultats

| Méthode | PSNR (dB) | SSIM | MSE | MAE |
|---------|-----------|------|-----|-----|
| classical | 21.54 ± 0.69 | 0.4129 ± 0.0158 | 4424.34 ± 266.71 | 50.38 ± 1.45 |
| ai_simple | 21.17 ± 0.42 | 0.8065 ± 0.0188 | 1900.43 ± 167.77 | 32.70 ± 1.89 |

## Analyse Détaillée

### Classical

- **Nombre de frames:** 30
- **Fichier vidéo:** data/output/demo_classical_restored.mp4

**Métriques détaillées:**

- **PSNR:**
  - Moyenne: 21.5380
  - Écart-type: 0.6878
  - Min: 20.5536
  - Max: 23.0576

- **SSIM:**
  - Moyenne: 0.4129
  - Écart-type: 0.0158
  - Min: 0.3918
  - Max: 0.4701

- **MSE:**
  - Moyenne: 4424.3429
  - Écart-type: 266.7059
  - Min: 4116.6160
  - Max: 5030.2611

- **MAE:**
  - Moyenne: 50.3762
  - Écart-type: 1.4548
  - Min: 48.8925
  - Max: 54.2533

### Ai_Simple

- **Nombre de frames:** 30
- **Fichier vidéo:** data/output/demo_ai_colorized.mp4

**Métriques détaillées:**

- **PSNR:**
  - Moyenne: 21.1690
  - Écart-type: 0.4232
  - Min: 20.5772
  - Max: 21.8228

- **SSIM:**
  - Moyenne: 0.8065
  - Écart-type: 0.0188
  - Min: 0.7290
  - Max: 0.8300

- **MSE:**
  - Moyenne: 1900.4298
  - Écart-type: 167.7656
  - Min: 1622.4350
  - Max: 2125.1030

- **MAE:**
  - Moyenne: 32.7030
  - Écart-type: 1.8851
  - Min: 29.7139
  - Max: 35.2457

## Interprétation

### PSNR (Peak Signal-to-Noise Ratio)
- Plus élevé = meilleure qualité
- > 30 dB: Bonne qualité
- > 40 dB: Très bonne qualité

### SSIM (Structural Similarity Index)
- Valeurs entre 0 et 1
- Plus proche de 1 = meilleure similitude structurelle
- > 0.9: Excellente similarité

### MSE/MAE
- Plus faible = meilleure qualité
- Mesure l'erreur pixel par pixel

