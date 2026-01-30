/**
 * @file activation.c
 * @brief Implémentation des fonctions d'activation
 *
 * Implémenter les fonctions déclarées dans nn/activation.h ici.
 *
 * NOTES D'IMPLÉMENTATION :
 *
 * Sigmoïde :
 *   sigma(x) = 1.0 / (1.0 + exp(-x))
 *   - Utiliser <math.h> pour exp()
 *   - Gérer le débordement : pour x très négatif, exp(-x) est énorme
 *     Limiter x à une plage raisonnable, par ex., [-500, 500]
 *
 * Dérivée de la sigmoïde :
 *   sigma'(x) = sigma(x) * (1 - sigma(x))
 *   Si vous avez output = sigma(x), alors : derivative = output * (1 - output)
 *
 * ReLU :
 *   relu(x) = x > 0 ? x : 0
 *
 * Dérivée de ReLU :
 *   relu'(x) = x > 0 ? 1 : 0
 */

#include "nn/nn_functions.h"
#include <math.h>

/* TODO : Implémenter les fonctions d'activation */
