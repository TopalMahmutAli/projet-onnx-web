# Reconnaissance de chiffres manuscrits (CNN → ONNX → Web)

Projet réalisé dans le cadre du cours IA pour le Web (IIM A4).
Objectif : entraîner un modèle CNN sur MNIST, l’exporter en ONNX, puis l’utiliser dans une interface web en HTML/CSS/JavaScript.

voici le liens githubpages: https://topalmahmutali.github.io/projet-onnx-web/

---

## Structure du projet

```
cours_ai_a4/
│
├── index.html
├── style.css
├── script.js
│
├── simple_cnn.onnx
├── simple_cnn.onnx.data
│
└── te.ipynb
```

---

## Modèle CNN (PyTorch)

Le modèle contient :

* deux couches convolutionnelles + ReLU
* max pooling
* dropout
* couches fully connected

Normalisation utilisée lors du training et du preprocessing web :

* mean = 0.1307
* std = 0.3081

---

## Entraînement

* Dataset : MNIST 28×28
* Optimizer : SGD
* Loss : CrossEntropy
* Tracking : TensorBoard
* Export : torch.onnx.export

---

## TensorBoard

<img width="573" height="523" alt="image" src="https://github.com/user-attachments/assets/3242408c-0dbb-46c3-9eab-2618b66cf498" />


---

## Export ONNX

Le modèle est exporté en deux fichiers :

* `simple_cnn.onnx` : graph du réseau
* `simple_cnn.onnx.data` : poids externes

Compatible avec onnxruntime-web.

---

## Interface Web

Le site permet de :

* dessiner un chiffre sur un canvas (blanc sur fond noir)
* appliquer un preprocessing identique au script Python
* charger le modèle ONNX dans le navigateur
* afficher la prédiction

La bibliothèque utilisée est :

```
onnxruntime-web
```
