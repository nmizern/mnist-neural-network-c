/**
 * @file mnist.c
 * @brief Implémentation du chargement du jeu de données MNIST
 *
 * Implémenter les fonctions déclarées dans nn/mnist.h ici.
 *
 * APPROCHE RECOMMANDÉE - Utilisation de stb_image.h :
 *
 * 1. Télécharger stb_image.h depuis :
 *    https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
 *
 * 2. Le placer dans include/third_party/stb_image.h
 *
 * 3. Dans UN fichier .c (celui-ci), faire :
 *    #define STB_IMAGE_IMPLEMENTATION
 *    #include "third_party/stb_image.h"
 *
 * 4. Utilisation :
 *    int width, height, channels;
 *    unsigned char *img = stbi_load(path, &width, &height, &channels, 1);
 *    // channels=1 force les niveaux de gris
 *    // img[i] est la valeur du pixel 0-255
 *    // Normaliser : float_pixel = img[i] / 255.0f
 *    stbi_image_free(img);
 *
 * ITÉRATION DE RÉPERTOIRE :
 * - Sur Windows : utiliser _findfirst/_findnext ou dirent.h (si disponible)
 * - Sur Linux/macOS : utiliser dirent.h (opendir, readdir)
 * - Multi-plateforme : envisager de générer la liste de fichiers en externe
 *
 * ALTERNATIVE SIMPLE :
 * Générer les chemins de fichiers programmatiquement :
 *   pour label 0-9 :
 *       pour i = 0 à count :
 *           sprintf(path, "%s/%d/%05d.png", base_path, label, i)
 *           charger l'image
 *
 * Note : Vous devez savoir combien d'images par classe ou détecter la fin des fichiers.
 */

#include "nn/mnist.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Décommenter après avoir ajouté stb_image.h dans include/third_party/ */
/* #define STB_IMAGE_IMPLEMENTATION */
/* #include "third_party/stb_image.h" */

/* TODO : Implémenter les fonctions de chargement MNIST */
