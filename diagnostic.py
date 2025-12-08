"""
Script de diagnostic rapide pour tester le modèle et identifier les problèmes
"""

import numpy as np
import tensorflow as tf
import os
import sys
from PIL import Image

def test_model_on_single_image(model_path, image_path):
    """Test le modèle sur une seule image"""
    print(f"🔍 Test du modèle sur une image unique")
    print(f"   Modèle: {model_path}")
    print(f"   Image: {image_path}")

    # Charger le modèle
    print("\n📥 Chargement du modèle...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"   ✅ Modèle chargé")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")

    # Charger et prétraiter l'image
    print("\n📥 Chargement de l'image...")
    img = Image.open(image_path).convert('RGB')
    print(f"   Taille originale: {img.size}")

    # Redimensionner à la taille attendue par le modèle
    target_size = model.input_shape[1:3]  # (height, width)
    img_resized = img.resize(target_size)
    print(f"   Taille redimensionnée: {img_resized.size}")

    # Convertir en array et normaliser
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)  # Ajouter dimension batch

    print(f"   Array shape: {img_batch.shape}")
    print(f"   Array min: {img_batch.min():.4f}")
    print(f"   Array max: {img_batch.max():.4f}")
    print(f"   Array mean: {img_batch.mean():.4f}")

    # Prédiction
    print("\n🔮 Prédiction...")
    pred = model.predict(img_batch, verbose=0)

    print(f"   Prediction shape: {pred.shape}")
    print(f"   Prediction value: {pred[0][0] if pred.shape[1] == 1 else pred[0]}")
    print(f"   Contains NaN: {np.isnan(pred).any()}")
    print(f"   Contains Inf: {np.isinf(pred).any()}")

    if np.isnan(pred).any():
        print("\n❌ PROBLÈME: La prédiction contient des NaN!")
        print("   Causes possibles:")
        print("   1. Le modèle a des poids NaN")
        print("   2. L'image contient des valeurs invalides")
        print("   3. Problème numérique dans le modèle")
    elif np.isinf(pred).any():
        print("\n❌ PROBLÈME: La prédiction contient des Inf!")
    else:
        print("\n✅ Prédiction OK")
        prob = float(pred[0][0] if pred.shape[1] == 1 else pred[0][0])
        print(f"   Probabilité de glaucome: {prob:.4%}")

    return pred


def test_model_weights(model_path):
    """Vérifie si le modèle a des poids NaN ou Inf"""
    print(f"\n🔍 Vérification des poids du modèle")

    model = tf.keras.models.load_model(model_path, compile=False)

    total_params = 0
    nan_params = 0
    inf_params = 0

    for layer in model.layers:
        weights = layer.get_weights()
        for w in weights:
            total_params += w.size
            nan_params += np.isnan(w).sum()
            inf_params += np.isinf(w).sum()

    print(f"   Total paramètres: {total_params:,}")
    print(f"   Paramètres NaN: {nan_params}")
    print(f"   Paramètres Inf: {inf_params}")

    if nan_params > 0:
        print("\n❌ PROBLÈME: Le modèle contient des poids NaN!")
        print("   Le modèle est corrompu ou n'a pas été entraîné correctement.")
        return False
    elif inf_params > 0:
        print("\n❌ PROBLÈME: Le modèle contient des poids Inf!")
        return False
    else:
        print("\n✅ Tous les poids sont valides")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 diagnostic.py <model_path> [image_path]")
        print("\nExemple:")
        print("  python3 diagnostic.py outputs/models/airogs_baseline_*.h5")
        print("  python3 diagnostic.py outputs/models/airogs_baseline_*.h5 dataset/5/image1.jpg")
        sys.exit(1)

    model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable: {model_path}")
        sys.exit(1)

    print("=" * 60)
    print("DIAGNOSTIC DU MODÈLE")
    print("=" * 60)

    # Test 1: Vérifier les poids
    weights_ok = test_model_weights(model_path)

    if not weights_ok:
        print("\n⚠️  Le modèle a des problèmes de poids. Arrêt du diagnostic.")
        sys.exit(1)

    # Test 2: Test sur une image (si fournie)
    if len(sys.argv) >= 3:
        image_path = sys.argv[2]
        if not os.path.exists(image_path):
            print(f"\n❌ Image introuvable: {image_path}")
        else:
            test_model_on_single_image(model_path, image_path)
    else:
        print("\n💡 Tip: Ajoutez un chemin d'image pour tester une prédiction:")
        print(f"   python3 diagnostic.py {model_path} dataset/5/image1.jpg")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()

