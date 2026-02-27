#include "nn/network.h"
#include "nn/mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Chemins par défaut relatifs à l'exécutable (build/tests/Release/) */
#define DEFAULT_DATA_PATH  "../../../data/test.bin"
#define DEFAULT_MODEL_PATH "../../../data/mnist_model.bin"

/* Affiche une image 28x28 en ASCII */
static void afficher_image(const float *pixels) {
    printf("\n");
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            float v = pixels[y * 28 + x];
            if      (v > 0.75f) printf("##");
            else if (v > 0.40f) printf("..");
            else                printf("  ");
        }
        printf("\n");
    }
    printf("\n");
}

/* Affiche les probabilités de sortie sous forme de barres */
static void afficher_probabilites(const Vector *output) {
    printf("Probabilites de sortie :\n");
    for (size_t i = 0; i < output->size; i++) {
        int bar = (int)(output->data[i] * 30.0f);
        printf("  %zu [", i);
        for (int b = 0; b < 30; b++) printf(b < bar ? "=" : " ");
        printf("] %.4f\n", output->data[i]);
    }
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    const char *data_path  = (argc > 1) ? argv[1] : DEFAULT_DATA_PATH;
    const char *model_path = (argc > 2) ? argv[2] : DEFAULT_MODEL_PATH;

    /* Demander un chiffre à l'utilisateur */
    int chiffre = -1;
    printf("Entrez un chiffre a reconnaitre (0-9) : ");
    if (scanf("%d", &chiffre) != 1 || chiffre < 0 || chiffre > 9) {
        fprintf(stderr, "Entree invalide. Veuillez entrer un chiffre entre 0 et 9.\n");
        return 1;
    }

    /* Charger le jeu de données de test */
    printf("Chargement du dataset depuis : %s\n", data_path);
    mnist_dataset_t *dataset = mnist_load_binary(data_path);
    if (!dataset) {
        return 1;
    }
    
    /* Collecter les indices correspondant au chiffre demandé */
    size_t *indices    = (size_t *)malloc(dataset->count * sizeof(size_t));
    size_t  nb_trouves = 0;
    for (size_t i = 0; i < dataset->count; i++) {
        if (dataset->labels[i] == (uint8_t)chiffre)
            indices[nb_trouves++] = i;
    }
    if (nb_trouves == 0) {
        fprintf(stderr, "Aucune image trouvee pour le chiffre %d.\n", chiffre);
        free(indices);
        mnist_free_dataset(dataset);
        return 1;
    }

    /* Choisir un échantillon aléatoire */
    size_t idx = indices[rand() % nb_trouves];
    free(indices);

    printf("Echantillon #%zu  (vrai label : %d)\n", idx, (int)dataset->labels[idx]);

    const float *pixels = &dataset->images[idx * MNIST_IMAGE_SIZE];
    afficher_image(pixels);

    /* Charger le réseau (architecture identique à mnist_train) */
    size_t layer_sizes[] = {784, 128, 10};
    Network *net = nn_network_create(layer_sizes, 3);
    printf("Chargement du modele depuis : %s\n", model_path);
    nn_network_load(net, model_path);

    /* Construire le vecteur d'entrée et lancer la prédiction */
    Vector *input = create_vector(MNIST_IMAGE_SIZE);
    for (size_t j = 0; j < MNIST_IMAGE_SIZE; j++)
        input->data[j] = pixels[j];

    size_t prediction;
    nn_network_predict(net, input, &prediction);

    afficher_probabilites(net->layers[net->num_layers - 1]->output_cache);

    printf("Vrai label  : %d\n", (int)dataset->labels[idx]);
    printf("Prediction  : %zu  %s\n",
           prediction,
           prediction == (size_t)dataset->labels[idx] ? "[CORRECT]" : "[FAUX]");

    free_vector(input);
    nn_network_free(net);
    mnist_free_dataset(dataset);
    return 0;
}
