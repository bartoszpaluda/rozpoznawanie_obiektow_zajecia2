# Szablon sprawozdania laboratoryjnego

**Przedmiot:** Uczenie głębokie do rozpoznawania obiektów  
**Zajęcia nr / temat:** Zajęcia 2 – Trenowanie i strojenie klasyfikatora obrazów  
**Autor:** Bartosz Paluda
**Data:** 27.10.2026  

---

## 1. Cel ćwiczenia
Celem ćwiczenia była poprawa jakości i zdolności generalizacji klasyfikatora obrazów opartego na perceptronie wielowarstwowym (MLP) na zbiorze FashionMNIST względem modelu bazowego (baseline) z Laboratorium 1. Zadanie obejmowało zbadanie wpływu pojedynczych oraz łączonych technik regularyzacji i optymalizacji: augmentacji danych, normalizacji wsadowej (BatchNorm), warstwy Dropout, kary wagowej (weight decay) oraz harmonogramu współczynnika uczenia (StepLR).

---

## 2. Metoda i implementacja
- **Zbiór danych i podział:** Wykorzystano zbiór FashionMNIST. Oryginalny zbiór treningowy podzielono na stały podzbiór treningowy (50 000 próbek) oraz stały podzbiór walidacyjny (10 000 próbek). Zbiór testowy (10 000 próbek) odseparowano i wykorzystano wyłącznie do ostatecznej ewaluacji wybranego modelu i baseline'u.
- **Augmentacja danych:** Zastosowano losowe odbicie poziome (`RandomHorizontalFlip`) oraz losowe przycięcie z marginesem (`RandomCrop(28, padding=4)`). Zgodnie z wytycznymi augmentacja była aplikowana wyłącznie na zbiorze treningowym; ewaluacja walidacyjna i testowa odbywała się na danych bez augmentacji.
- **Architektura modelu i regularyzacja:** Szkielet modelu oparto na sieci MLP o warstwach ukrytych (256, 128). Warianty regularyzacyjne obejmowały dodanie warstw `BatchNorm1d` oraz `Dropout(p=0.3)` po funkcjach aktywacji ReLU.
- **Trening i ewaluacja:** Do optymalizacji wykorzystano algorytm Adam z funkcją straty CrossEntropyLoss. W wariancie strojonym włączono weight decay (0.0001 / 1e-4) oraz scheduler `StepLR` (zmniejszający LR o parametr `gamma=0.5`).
- **Luka generalizacji:** W celu rzetelnej oceny luki train-validation w każdej epoce mierzono stratę walidacyjną oraz czystą stratę treningową na danych bez augmentacji w trybie `model.eval()`.
- **Zadania dodatkowe:** 
  1. Zaimplementowano mechanizm wczesnego zatrzymania (*early stopping* z cierpliwością 4 epok), który po zakończeniu treningu przywraca wagi z checkpointu o najniższej stracie walidacyjnej.
  2. Przeprowadzono przegląd współczynnika uczenia dla wartości LR z zakresu {0.1, 0.01, 0.001, 0.0001} na przestrzeni 4 epok.

- **Link do Github'a:** https://github.com/bartoszpaluda/rozpoznawanie_obiektow_zajecia2/tree/main
---

## 3. Wyniki

Tabela 1. Zestawienie przetestowanych wariantów konfiguracji (selekcja według najniższej straty walidacyjnej)]:

| Konfiguracja | Wybrana epoka | Accuracy walidacyjna [%] | Luka train-validation (val_loss - train_eval_loss) |
| :--- | :---: | :---: | :---: |
| **Baseline** (brak regularyzacji, dane czyste) | 7 | 89.13% | 0.1072 |
| **Wariant 1** (Augmentacja danych) | 12 | 83.35% | 0.0203 |
| **Wariant 2** (BatchNorm + Dropout 0.3) | 11 | 89.24% | 0.0947 |
| **Wariant strojony** (Aug + BN + Dropout + WD + StepLR) | 12 | 82.36% | 0.0136 |

Końcowa ocena testowa (po zamrożeniu decyzji architektonicznych):
- **Końcowe accuracy testowe (Baseline):** 88.29%
- **Końcowe accuracy testowe (Wariant strojony):** 81.60%

Przegląd współczynnika uczenia (LR vs accuracy walidacyjna po 4 epokach):
- LR = 0.1 → 20.07%
- LR = 0.01 → 86.75%
- LR = 0.001 → 88.32%
- LR = 0.0001 → 85.64%
  
<img width="699" height="393" alt="baseline" src="https://github.com/user-attachments/assets/ab532804-6a95-42c4-af1f-be1b7eb8ffd2" />


<img width="691" height="393" alt="wariant" src="https://github.com/user-attachments/assets/3b582395-0005-44dd-842f-b81dc5f593bd" />


<img width="531" height="398" alt="wplyw" src="https://github.com/user-attachments/assets/d713830c-1315-49ac-80e8-649cd16909eb" />


---

## 4. Analiza i interpretacja
- **Luka generalizacji i przeuczenie w baseline:** Model bazowy osiągnął wysoką dokładność na zbiorze walidacyjnym (89.13%), jednak charakteryzował się wyraźną luką generalizacji wynoszącą 0.1072. Krzywa straty walidacyjnej po 7. epoce przestała spadać, podczas gdy czysta strata treningowa stale malała, co jest podręcznikowym objawem przeuczenia (overfittingu).
- **Wpływ technik regularyzacyjnych:** Zastosowanie połączonej regularyzacji (Wariant strojony) doprowadziło do niemal całkowitej redukcji luki generalizacji – spadła ona z 0.1072 do 0.0136. Krzywe straty treningowej i walidacyjnej przebiegały bardzo blisko siebie.
- **Zjawisko niedouczenia (underfitting):** Pomimo minimalnej luki generalizacji wariant strojony osiągnął niższe accuracy testowe (81.60%) niż baseline (88.29%). Zgodnie z wytycznymi laboratoryjnymi bardzo mała luka przy zauważalnym spadku dokładności świadczy o **niedouczeniu modelu**. Sieć typu MLP spłaszcza dwuwymiarowy obraz do jednowymiarowego wektora i nie posiada niezmienniczości przesunięć cech charakterystycznych dla sieci konwolucyjnych (CNN). W efekcie jednoczesne nałożenie augmentacji przestrzennej (`RandomCrop`, `RandomHorizontalFlip`) oraz warstwy Dropout o współczynniku 0.3 stanowiło dla prostej architektury MLP zbyt restrykcyjną barierę pojemnościową.
- **Analiza współczynnika uczenia (LR):** Eksperyment potwierdził, że LR = 0.001 jest optymalną wartością dla optymalizatora Adam. Zbyt duży współczynnik LR = 0.1 uniemożliwił zbieżność procesu optymalizacji (accuracy zaledwie 20.07%), natomiast zbyt mały LR = 0.0001 spowolnił tempo uczenia.

---

## 5. Wnioski
1. Metody regularyzacyjne (augmentacja, dropout, weight decay) skutecznie eliminują przeuczenie i domykają lukę generalizacji między zbiorem treningowym a walidacyjnym.
2. Nadmierna regularyzacja modeli o ograniczonej pojemności reprezentacyjnej (jak MLP na danych obrazowych) prowadzi do zjawiska niedouczenia (underfittingu), obniżając ogólną zdolność predykcyjną sieci.
3. Mechanizm *early stopping* efektywnie zabezpiecza model przed degradacją wag w późniejszych epokach, zatrzymując stan parametrów z punktu o najniższej stracie walidacyjnej.
4. Prawidłowy dobór współczynnika uczenia jest kluczowym hiperparametrem – skrajne wartości uniemożliwiają poprawną zbieżność sieci.
