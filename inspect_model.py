"""
Script pour inspecter en détail un modèle et identifier les problèmes
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import os

def inspect_model(model_path):
    """Inspecte un modèle en détail"""

    print("=" * 70)
    print("INSPECTION DÉTAILLÉE DU MODÈLE")
    print("=" * 70)
    print(f"\n📁 Modèle: {model_path}\n")

    # 1. Informations sur le fichier
    print("📊 Informations Fichier:")
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   Taille: {file_size:.2f} MB")

    print("\n📥 Chargement du modèle...")
    try:
        model = keras.models.load_model(model_path, compile=False)
        print("   ✅ Modèle chargé avec succès")
    except Exception as e:
        print(f"   ❌ Erreur lors du chargement: {e}")
        print("   ⚙️  Tentative de chargement alternatif depuis le fichier HDF5...")

        try:
            import h5py
            import json

            with h5py.File(model_path, 'r') as f:
                # Cas classique: Keras HDF5 stocke la config sous l'attribut 'model_config'
                if 'model_config' in f.attrs:
                    raw = f.attrs['model_config']
                    if isinstance(raw, bytes):
                        raw = raw.decode('utf-8')
                    model_config = json.loads(raw)
                    model = keras.models.model_from_config(model_config, custom_objects=None)
                    # Charger les poids depuis le fichier HDF5
                    model.load_weights(model_path)
                    print("   ✅ Modèle reconstruit depuis `model_config` et poids chargés")
                else:
                    raise RuntimeError("`model_config` introuvable dans le fichier HDF5.")

        except Exception as e2:
            print(f"   ❌ Échec du chargement alternatif: {e2}")
            print("   🔎 Diagnostics rapides:")
            print(
                "     - Vérifier que la version de TensorFlow/Keras utilisée pour l'inspection est la même que celle utilisée pour l'entraînement.")
            print(
                "     - Si le modèle contient des custom layers/activations, passez-les via `custom_objects` à `load_model`.")
            print("     - Exemple: keras.models.load_model(path, custom_objects={'MaLayer': MaLayer})")
            print(
                "     - Si rien ne marche, extraire les poids (h5py) et reconstruire manuellement l'architecture avant d'appeler `load_weights`.")
            return False

    # 3. Architecture
    print("\n🏗️  Architecture:")
    print(f"   Nombre de couches: {len(model.layers)}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")

    # 4. Dernières couches (importantes pour la classification)
    print("\n🔍 Dernières couches:")
    for i, layer in enumerate(model.layers[-5:]):
        print(f"   {len(model.layers)-5+i}. {layer.name} ({layer.__class__.__name__})")
        if hasattr(layer, 'activation'):
            print(f"      Activation: {layer.activation.__name__ if hasattr(layer.activation, '__name__') else layer.activation}")
        if hasattr(layer, 'units'):
            print(f"      Units: {layer.units}")

    # 5. Vérifier la couche de sortie
    output_layer = model.layers[-1]
    print(f"\n🎯 Couche de Sortie:")
    print(f"   Type: {output_layer.__class__.__name__}")
    print(f"   Nom: {output_layer.name}")

    if hasattr(output_layer, 'activation'):
        activation = output_layer.activation
        activation_name = activation.__name__ if hasattr(activation, '__name__') else str(activation)
        print(f"   Activation: {activation_name}")

        # Vérifier si l'activation est appropriée
        if 'sigmoid' not in activation_name.lower():
            print(f"   ⚠️  WARNING: L'activation de sortie devrait être 'sigmoid' pour une classification binaire!")

    if hasattr(output_layer, 'units'):
        print(f"   Units: {output_layer.units}")
        if output_layer.units != 1:
            print(f"   ⚠️  WARNING: Pour une classification binaire, la sortie devrait avoir 1 unité!")

    # 6. Statistiques des poids
    print("\n⚖️  Statistiques des Poids:")
    total_params = 0
    trainable_params = 0
    nan_params = 0
    zero_params = 0

    for layer in model.layers:
        weights = layer.get_weights()
        for w in weights:
            total_params += w.size
            nan_params += np.isnan(w).sum()
            zero_count = (w == 0).sum()
            zero_params += zero_count

        if layer.trainable:
            for w in layer.get_weights():
                trainable_params += w.size

    print(f"   Total paramètres: {total_params:,}")
    print(f"   Paramètres entraînables: {trainable_params:,}")
    print(f"   Paramètres NaN: {nan_params:,}")
    print(f"   Paramètres à zéro: {zero_params:,} ({100*zero_params/total_params:.2f}%)")

    if nan_params > 0:
        print(f"\n   ❌ CRITIQUE: Le modèle contient {nan_params} poids NaN!")
        return False

    if zero_params == total_params:
        print(f"\n   ❌ CRITIQUE: TOUS les poids sont à zéro! Le modèle n'a pas été entraîné!")
        return False

    if zero_params > total_params * 0.9:
        print(f"\n   ⚠️  WARNING: Plus de 90% des poids sont à zéro!")

    # 7. Distribution des poids de la dernière couche
    print("\n📊 Distribution Poids Couche de Sortie:")
    output_weights = output_layer.get_weights()
    if len(output_weights) > 0:
        weights = output_weights[0]
        print(f"   Shape: {weights.shape}")
        print(f"   Min: {weights.min():.6f}")
        print(f"   Max: {weights.max():.6f}")
        print(f"   Mean: {weights.mean():.6f}")
        print(f"   Std: {weights.std():.6f}")

        if len(output_weights) > 1:
            bias = output_weights[1]
            print(f"\n   Bias shape: {bias.shape}")
            print(f"   Bias values: {bias}")
    else:
        print(f"   ⚠️  Pas de poids dans la couche de sortie!")

    # 8. Test avec une image synthétique
    print("\n🧪 Test avec image synthétique:")

    # Créer une image de test (bruit aléatoire)
    input_shape = model.input_shape[1:]  # Sans la dimension batch
    test_image = np.random.rand(1, *input_shape).astype(np.float32)

    print(f"   Image test shape: {test_image.shape}")
    print(f"   Image test min/max: {test_image.min():.3f} / {test_image.max():.3f}")

    try:
        pred = model.predict(test_image, verbose=0)
        print(f"\n   Prédiction shape: {pred.shape}")
        print(f"   Prédiction value: {pred[0]}")
        print(f"   Contains NaN: {np.isnan(pred).any()}")
        print(f"   Contains Inf: {np.isinf(pred).any()}")

        if np.isnan(pred).any():
            print(f"\n   ❌ La prédiction contient des NaN!")
            return False

        if pred[0][0] == 0.0:
            print(f"\n   ⚠️  WARNING: La prédiction est exactement 0.0!")
            print(f"   Cela suggère un problème dans le modèle.")

    except Exception as e:
        print(f"\n   ❌ Erreur lors de la prédiction: {e}")
        return False

    # 9. Test avec une image noire et une blanche
    print("\n🧪 Test avec images extrêmes:")

    # Image noire
    black_image = np.zeros((1, *input_shape), dtype=np.float32)
    pred_black = model.predict(black_image, verbose=0)
    print(f"   Image noire → Prédiction: {pred_black[0][0]:.6f}")

    # Image blanche
    white_image = np.ones((1, *input_shape), dtype=np.float32)
    pred_white = model.predict(white_image, verbose=0)
    print(f"   Image blanche → Prédiction: {pred_white[0][0]:.6f}")

    if pred_black[0][0] == pred_white[0][0]:
        print(f"\n   ⚠️  WARNING: Même prédiction pour image noire et blanche!")
        print(f"   Le modèle ne répond pas aux variations d'entrée.")

    # 10. Résumé
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ")
    print("=" * 70)

    issues = []

    if nan_params > 0:
        issues.append("❌ Poids NaN détectés - Modèle corrompu")

    if zero_params == total_params:
        issues.append("❌ Tous les poids sont à zéro - Modèle non entraîné")

    if pred_black[0][0] == pred_white[0][0] == 0.0:
        issues.append("❌ Prédictions constantes à 0.0 - Modèle non fonctionnel")

    if hasattr(output_layer, 'activation'):
        activation_name = output_layer.activation.__name__ if hasattr(output_layer.activation, '__name__') else str(output_layer.activation)
        if 'sigmoid' not in activation_name.lower():
            issues.append(f"⚠️  Activation de sortie '{activation_name}' au lieu de 'sigmoid'")

    if issues:
        print("\n🔴 PROBLÈMES DÉTECTÉS:")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 RECOMMANDATIONS:")
        print("   1. Vérifiez les logs d'entraînement")
        print("   2. Assurez-vous que l'entraînement s'est terminé correctement")
        print("   3. Vérifiez que vous chargez le bon fichier de modèle")
        print("   4. Ré-entraînez le modèle si nécessaire")
        return False
    else:
        print("\n✅ Le modèle semble OK")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_model.py <model_path>")
        print("\nExemple:")
        print("  python3 inspect_model.py outputs/models/airogs_baseline_*.h5")
        sys.exit(1)

    model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"❌ Fichier introuvable: {model_path}")
        sys.exit(1)

    success = inspect_model(model_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

