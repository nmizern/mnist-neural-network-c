[English](./README.md) | [**Français**](./README.fr.md)

# Reseau de neurones en C

Bibliotheque C de reseau de neurones dense (fully connected) pour la classification MNIST, avec interface web en Go.

https://github.com/nmizern/mnist-neural-network-c

**Auteurs:** Mikita Mizerkin, Idirene Daris

**Cours:** TEI S7 - Projet Reseaux de neurones

> **Projet soutenu avec succes devant la commission - note 19/20, meilleur resultat de la promotion L3 E3A.**
>
> Developpe sur 10 seances de 4 heures, completees par du travail personnel a la maison.

![Screenshot de l'interface web](./docs/image.png)

## Prerequis

- Compilateur C (GCC, Clang ou MSVC)
- CMake 3.12+
- libpng

```bash
# Ubuntu / Debian
sudo apt-get install libpng-dev

# macOS
brew install libpng

# Windows (vcpkg)
vcpkg install libpng
```

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

Pour un build optimise:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Tests

```bash
cd build
ctest --output-on-failure
```

## Entrainement

Le repository inclut les donnees pre-traitees (`examples/mnist-pngs/test.bin`, `mnist_model.bin`). Pour re-entrainer depuis zero, il faut le dataset PNG complet:

```bash
git clone https://github.com/rasbt/mnist-pngs examples/mnist-pngs
```

Lancer l'entrainement:

```bash
./build/examples/mnist_train examples/mnist-pngs
```

Plusieurs configurations d'entrainement sont disponibles dans `examples/`:

| Exemple | Activation | Perte | Optimiseur |
|---------|-----------|-------|------------|
| `mnist_train.c` | Sigmoid | MSE | SGD |
| `mnist_train_sigmoid_sgd_mse.c` | Sigmoid | MSE | SGD |
| `mnist_train_relu_softmax_adam_cce.c` | ReLU / Softmax | Cross-Entropy | Adam |
| `mnist_train_tanh_softmax_sgd_cce.c` | Tanh / Softmax | Cross-Entropy | SGD |
| `mnist_train_architectures.c` | Sigmoid | MSE | SGD (compare plusieurs architectures) |

## Interface web

Application Go avec Gin pour dessiner et reconnaitre des chiffres en temps reel.

```bash
cd web
go mod download
go run . -model ../examples/mnist-pngs/mnist_model.bin -addr :4343
```

Ouvrir `http://localhost:4343` dans un navigateur.

Le serveur charge le modele binaire au demarrage et effectue la propagation avant en Go pur (sans cgo). L'interface permet de dessiner un chiffre sur un canvas, le preprocessing (centrage, mise a l'echelle 28x28) est fait cote client avant l'envoi au serveur.

## Utilisation de la bibliotheque

```c
#include "neuralnet.h"

const size_t layers[] = {784, 256, 128, 10};
Network *net = nn_network_create(layers, 4);

// Configuration de l'entrainement
nn_network_set_training_config(net,
                               ACTIVATION_RELU,
                               ACTIVATION_SOFTMAX,
                               LOSS_CATEGORICAL_CROSS_ENTROPY,
                               OPTIMIZER_ADAM);

nn_network_train(net, train_data, 0.001f, 10);
float acc = nn_network_evaluate(net, test_data);
nn_network_save(net, "model.bin");
nn_network_free(net);
```

Activations disponibles: `SIGMOID`, `RELU`, `SOFTMAX`, `TANH`, `LEAKY_RELU`, `LINEAR`

Fonctions de perte: `MSE`, `MAE`, `BINARY_CROSS_ENTROPY`, `CATEGORICAL_CROSS_ENTROPY`

Optimiseurs: `SGD`, `ADAM`

## Structure du projet

```
nn/
├── include/
│   ├── neuralnet.h            # en-tete principal
│   └── nn/
│       ├── matrix.h           # matrices et vecteurs
│       ├── mnist.h            # chargement MNIST
│       ├── network.h          # reseau de neurones
│       └── nn_functions.h     # activations, pertes, optimiseurs
├── src/                       # implementation
├── examples/                  # exemples d'entrainement
├── tests/                     # tests unitaires
├── web/                       # interface web Go
│   ├── main.go               # serveur Gin
│   ├── nn/network.go         # inference en Go
│   ├── templates/index.html  # interface canvas
│   └── static/app.js         # logique client
├── CMakeLists.txt
└── CMakePresets.json
```

## Notes techniques

- Le chargement MNIST utilise un cache binaire (`train.bin`/`test.bin`) pour accelerer les relances (quelques secondes au lieu de plusieurs minutes pour 70K PNG).
- L'entrainement fonctionne sans allocation memoire dynamique pendant les passes avant/arriere (buffers pre-alloues).
- L'interface web utilise un centrage par centre de masse pour le preprocessing des chiffres dessines, reproduisant le preprocessing MNIST original.

## Licence

Projet universitaire - TEI S7.
