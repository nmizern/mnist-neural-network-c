# Reseau de neurones en C

Bibliotheque C de reseau de neurones dense (fully connected) orientee classification MNIST.

https://gitlab.com/MadebyDaris/nn

**Auteurs:** Mikita Mizerkin, Idirene Daris  
**Cours:** TEI S6 - Projet Reseaux de neurones

Example de fonctionnement de l'application

![Screenshot du déliverable](./docs/image.png)

## Objectif

Le projet fournit une implementation from scratch en C pour:

1. Definir des architectures multicouches (taille variable)
2. Executer la propagation avant (inference)
3. Entrainer par retropropagation
4. Evaluer sur MNIST

## Prerequis

- Compilateur C (GCC, Clang ou MSVC)
- CMake 3.12+
- libpng (lecture des images PNG MNIST)

Exemples d'installation de libpng:

```bash
# Ubuntu / Debian
sudo apt-get install libpng-dev

# macOS
brew install libpng

# Windows (vcpkg)
vcpkg install libpng
```

## Build rapide

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Lancer les tests

```bash
ctest --output-on-failure
```

## Lancer un entrainement MNIST

1. Recuperer le dataset PNG:

```bash
git clone https://github.com/rasbt/mnist-pngs data/mnist-pngs
```

1. Lancer un exemple:

```bash
./examples/mnist_train ../data/mnist-pngs
```

## Structure du projet

```text
nn/
├── include/
│   ├── neuralnet.h
│   └── nn/
│       ├── matrix.h
│       ├── mnist.h
│       ├── network.h
│       └── nn_functions.h
├── src/
│   ├── matrix.c
│   ├── mnist.c
│   ├── network.c
│   └── nn_functions.c
├── tests/
│   └── test_*.c
├── examples/
│   ├── mnist_train.c
│   ├── mnist_train_architectures.c
│   ├── mnist_train_sigmoid_sgd_mse.c
│   ├── mnist_train_relu_softmax_adam_cce.c
│   └── mnist_train_tanh_softmax_sgd_cce.c
├── web/
│   ├── main.go
│   ├── nn/network.go
│   ├── templates/index.html
│   └── static/app.js
├── docs/
├── CMakeLists.txt
└── CMakePresets.json
```

## Structure d'un reseau (dans ce projet)

Un reseau est defini par un tableau de tailles de couches, par exemple `{784, 256, 128, 10}`:

- `784`: couche d'entree (image MNIST 28x28 aplatie)
- `256, 128`: couches cachees
- `10`: couche de sortie (classes 0 a 9)

Dans l'implementation:

- chaque couche stocke une matrice de poids `W` avec biais integre (colonne supplementaire)
- la propagation avant applique `W * [x | 1]`, puis la fonction d'activation choisie
- la retropropagation calcule les deltas et met a jour les poids via l'optimiseur configure (SGD ou Adam)

La configuration d'entrainement est parametree dans le reseau:

- activation cachee
- activation de sortie
- fonction de perte
- type d'optimiseur

## Utiliser la librairie pour creer et entrainer un reseau

Le point d'entree C est l'en-tete `include/neuralnet.h`.

Exemple minimal:

```c
#include "neuralnet.h"

const size_t layers[] = {784, 256, 128, 10};
Network *net = nn_network_create(layers, 4);

nn_network_set_training_config(net,
                               ACTIVATION_RELU,
                               ACTIVATION_SOFTMAX,
                               LOSS_CATEGORICAL_CROSS_ENTROPY,
                               OPTIMIZER_ADAM);

nn_network_train(net, train_data, 0.001f, 10);
float acc = nn_network_evaluate(net, test_data);
nn_network_save(net, "mnist_model.bin");
nn_network_free(net);
```

Cycle typique d'utilisation:

1. Charger les donnees MNIST (`mnist_load_dataset` ou cache binaire).
2. Creer le reseau avec `nn_network_create`.
3. Choisir la config d'entrainement avec `nn_network_set_training_config`.
4. Entrainer avec `nn_network_train`.
5. Evaluer avec `nn_network_evaluate`.
6. Sauvegarder/charger avec `nn_network_save` et `nn_network_load`.

Pour des cas concrets, voir les programmes dans `examples/`.

## Comment le web marche

La partie web est une application Go independante dans `web/`.

Architecture:

1. `web/main.go` charge un modele binaire (`mnist_model.bin`) au demarrage.
2. Le serveur Gin expose:
   - `GET /` pour la page HTML
   - `POST /api/predict` pour la prediction
3. Le front (`templates/index.html` + `static/app.js`) permet de dessiner un chiffre sur un canvas.
4. Le JavaScript:
   - extrait le dessin
   - recadre + recentre le chiffre
   - redimensionne en 28x28
   - normalise les pixels entre 0 et 1
   - envoie 784 valeurs a `/api/predict`
5. Le backend Go calcule la propagation avant et renvoie:
   - le chiffre predit (`digit`)
   - le vecteur de confiance (`confidence`)

Lancer le web:

```bash
cd web
go run . -model ../data/mnist_model.bin -addr :4343
```

Puis ouvrir `http://localhost:4343`.

Note de compatibilite:

- le lecteur de modele Go applique actuellement une activation sigmoide sur chaque couche lors de la prediction.
- pour des resultats coherents, utilisez de preference un modele entraine avec une configuration proche (ex: Sigmoid + SGD/MSE), ou adaptez `web/nn/network.go` pour refléter exactement la config d'entrainement du modele.

## Detail des dossiers

- include/: API publique du projet.
- src/: implementation des modules (matrices, fonctions NN, reseau, loader MNIST).
- tests/: tests unitaires de validation fonctionnelle.
- examples/: executables de demonstration et de comparaison d'architectures/configurations.
- web/: interface de dessin/prediction et serveur HTTP Go.
- docs/: notes d'implementation et documentation du projet.

## Variantes MNIST disponibles

- mnist_train.c: baseline historique.
- mnist_train_architectures.c: comparaison de plusieurs architectures.
- mnist_train_sigmoid_sgd_mse.c: Sigmoid + MSE + SGD.
- mnist_train_relu_softmax_adam_cce.c: ReLU + Softmax + CCE + Adam.
- mnist_train_tanh_softmax_sgd_cce.c: TANH + Softmax + CCE + SGD.

## Notes techniques

- Le chargement MNIST utilise un cache binaire (train.bin/test.bin) pour accelerer les relances.
- Le module reseau permet maintenant de configurer activation, loss et optimiseur via l'API.

## Licence

Projet universitaire - TEI S7.
