/**
 * @file xor_example.c
 * @brief Problème XOR - test simple pour réseau de neurones
 *
 * Le problème XOR est un test classique pour les réseaux de neurones.
 * Il nécessite au moins une couche cachée car XOR n'est pas linéairement séparable.
 *
 * Table de vérité XOR :
 *   Entrée   | Sortie
 *   0, 0     | 0
 *   0, 1     | 1
 *   1, 0     | 1
 *   1, 1     | 0
 *
 * RÉSEAU RECOMMANDÉ :
 * - Entrée : 2 neurones
 * - Caché : 4-8 neurones avec Sigmoïde ou ReLU
 * - Sortie : 1 neurone avec Sigmoïde
 *
 * Ceci est un bon premier test avant d'aborder MNIST :
 * 1. Créer un réseau avec les tailles [2, 4, 1]
 * 2. Entraîner sur 4 échantillons XOR
 * 3. Tester les prédictions
 *
 * Si XOR fonctionne, votre propagation avant/arrière est probablement correcte.
 */

#include "neuralnet.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("XOR Neural Network Example\n");
    printf("==========================\n\n");

    /* TODO : Implémenter l'exemple d'entraînement XOR
     *
     * Étapes :
     * 1. Définir les données d'entraînement XOR
     *    float inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
     *    float targets[4] = {0, 1, 1, 0};
     *
     * 2. Créer le réseau : [2, 4, 1]
     *
     * 3. Entraîner pour ~10000 époques avec learning_rate ~0.5
     *
     * 4. Tester les prédictions et afficher les résultats
     */

    printf("TODO: Implement XOR example\n");

    return 0;
}
