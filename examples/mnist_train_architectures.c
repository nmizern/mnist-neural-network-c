#include "neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    const char *name;
    const size_t *layers;
    size_t num_layers;
    float learning_rate;
    int epochs;
} RunConfig;

static mnist_dataset_t *load_or_build_cache(const char *data_path, const char *split) {
    // Utilise d'abord le cache binaire pour accelerer les executions suivantes.
    char bin_path[512];
    snprintf(bin_path, sizeof(bin_path), "%s/%s.bin", data_path, split);

    mnist_dataset_t *dataset = mnist_load_binary(bin_path);
    if (dataset) {
        printf("Charge %zu echantillons %s depuis le cache (%s)\n", dataset->count, split, bin_path);
        return dataset;
    }

    dataset = mnist_load_dataset(data_path, split);
    if (!dataset) {
        fprintf(stderr, "Echec du chargement du jeu %s\n", split);
        return NULL;
    }

    printf("Charge %zu echantillons %s depuis les dossiers PNG\n", dataset->count, split);
    mnist_save_binary(dataset, bin_path);
    printf("Cache sauvegarde: %s\n", bin_path);
    return dataset;
}

static void run_single_config(const RunConfig *cfg, mnist_dataset_t *train, mnist_dataset_t *test) {
    // Cree un reseau pour cette architecture puis entraine/evalue avec ses hyperparametres.
    Network *network = nn_network_create(cfg->layers, cfg->num_layers);
    if (!network) {
        fprintf(stderr, "Impossible de creer le reseau pour la config: %s\n", cfg->name);
        return;
    }

    printf("\n=== %s ===\n", cfg->name);
    nn_network_print(network);
    printf("Entrainement avec lr=%.4f pendant %d epochs...\n", cfg->learning_rate, cfg->epochs);
    nn_network_train(network, train, cfg->learning_rate, cfg->epochs);

    float accuracy = nn_network_evaluate(network, test);
    printf("Precision test (%s): %.2f%%\n", cfg->name, accuracy * 100.0f);

    nn_network_free(network);
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    if (argc < 2) {
        printf("Usage: %s <chemin_vers_racine_mnist_pngs>\n", argv[0]);
        printf("Dossiers attendus:\n");
        printf("  train/0..9 et test/0..9\n");
        return 1;
    }

    const char *data_path = argv[1];
    mnist_dataset_t *train = load_or_build_cache(data_path, "train");
    if (!train) {
        return 1;
    }

    mnist_dataset_t *test = load_or_build_cache(data_path, "test");
    if (!test) {
        mnist_free_dataset(train);
        return 1;
    }

    const size_t arch_small[] = {784, 64, 10};
    const size_t arch_baseline[] = {784, 128, 64, 10};
    const size_t arch_wide[] = {784, 256, 128, 10};
    const size_t arch_deep[] = {784, 256, 128, 64, 10};

    const RunConfig runs[] = {
        {"small_64", arch_small, sizeof(arch_small) / sizeof(arch_small[0]), 0.10f, 5},
        {"baseline_128_64", arch_baseline, sizeof(arch_baseline) / sizeof(arch_baseline[0]), 0.08f, 6},
        {"wide_256_128", arch_wide, sizeof(arch_wide) / sizeof(arch_wide[0]), 0.05f, 6},
        {"deep_256_128_64", arch_deep, sizeof(arch_deep) / sizeof(arch_deep[0]), 0.04f, 8}
    };

    // Lance les architectures dans un ordre fixe pour comparer les resultats.
    const size_t run_count = sizeof(runs) / sizeof(runs[0]);
    for (size_t i = 0; i < run_count; i++) {
        run_single_config(&runs[i], train, test);
    }

    mnist_free_dataset(train);
    mnist_free_dataset(test);
    return 0;
}
