"""
Script de test rapide pour vérifier que l'entraînement et le chargement fonctionnent
"""

import tensorflow as tf
from model import create_baseline_model, compile_model
import numpy as np
import os

print("=" * 70)
print("TEST RAPIDE: Entraînement et Chargement de Modèle")
print("=" * 70)

# 1. Créer un modèle
print("\n1️⃣  Création du modèle...")
model = create_baseline_model(backbone="efficientnet-b0", input_shape=(384, 384, 3))
print(f"   ✅ Modèle créé")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Total params: {model.count_params():,}")

# 2. Compiler le modèle
print("\n2️⃣  Compilation du modèle...")
model = compile_model(model, learning_rate=1e-4)
print(f"   ✅ Modèle compilé")

# 3. Test avec données synthétiques
print("\n3️⃣  Génération de données synthétiques...")
n_samples = 100
X_train = np.random.rand(n_samples, 384, 384, 3).astype(np.float32)
y_train = np.random.randint(0, 2, (n_samples,)).astype(np.float32)
print(f"   ✅ {n_samples} échantillons générés")
print(f"   X shape: {X_train.shape}")
print(f"   y shape: {y_train.shape}")
print(f"   y distribution: {np.bincount(y_train.astype(int))}")

# 4. Prédiction avant entraînement
print("\n4️⃣  Prédiction AVANT entraînement (poids aléatoires)...")
preds_before = model.predict(X_train[:10], verbose=0)
print(f"   Predictions shape: {preds_before.shape}")
print(f"   Min: {preds_before.min():.4f}")
print(f"   Max: {preds_before.max():.4f}")
print(f"   Mean: {preds_before.mean():.4f}")
print(f"   Sample predictions: {preds_before[:5].flatten()}")

# 5. Entraîner 1 époque
print("\n5️⃣  Entraînement (1 époque)...")
history = model.fit(X_train, y_train, epochs=1, batch_size=8, verbose=1, validation_split=0.2)

print(f"\n   Métriques après entraînement:")
print(f"   Loss: {history.history['loss'][0]:.4f}")
print(f"   AUC: {history.history['auc'][0]:.4f}")

# 6. Prédiction après entraînement
print("\n6️⃣  Prédiction APRÈS entraînement...")
preds_after = model.predict(X_train[:10], verbose=0)
print(f"   Predictions shape: {preds_after.shape}")
print(f"   Min: {preds_after.min():.4f}")
print(f"   Max: {preds_after.max():.4f}")
print(f"   Mean: {preds_after.mean():.4f}")
print(f"   Sample predictions: {preds_after[:5].flatten()}")

# Vérifier que les prédictions ont changé
diff = np.abs(preds_after - preds_before).mean()
print(f"\n   Différence moyenne avant/après: {diff:.4f}")
if diff < 0.001:
    print(f"   ⚠️  WARNING: Les prédictions n'ont presque pas changé!")
else:
    print(f"   ✅ Les prédictions ont changé (normal après entraînement)")

# 7. Sauvegarder
print("\n7️⃣  Sauvegarde du modèle...")
test_model_path = "test_model_temp.h5"
model.save(test_model_path)
file_size = os.path.getsize(test_model_path) / (1024 * 1024)
print(f"   ✅ Modèle sauvegardé: {test_model_path}")
print(f"   Taille: {file_size:.2f} MB")

# 8. Charger le modèle
print("\n8️⃣  Chargement du modèle...")
loaded_model = tf.keras.models.load_model(test_model_path, compile=False)
print(f"   ✅ Modèle chargé")

# 9. Prédiction avec modèle chargé
print("\n9️⃣  Prédiction avec modèle chargé...")
X_test = np.random.rand(10, 384, 384, 3).astype(np.float32)
preds_loaded = loaded_model.predict(X_test, verbose=0)

print(f"   Predictions shape: {preds_loaded.shape}")
print(f"   Min: {preds_loaded.min():.4f}")
print(f"   Max: {preds_loaded.max():.4f}")
print(f"   Mean: {preds_loaded.mean():.4f}")
print(f"   Sample predictions: {preds_loaded[:5].flatten()}")

# 10. Vérifications finales
print("\n" + "=" * 70)
print("📋 VÉRIFICATIONS FINALES")
print("=" * 70)

issues = []

if preds_loaded.min() == preds_loaded.max() == 0.0:
    issues.append("❌ CRITIQUE: Toutes les prédictions sont 0.0!")

if np.isnan(preds_loaded).any():
    issues.append("❌ CRITIQUE: Prédictions contiennent des NaN!")

if np.isinf(preds_loaded).any():
    issues.append("❌ CRITIQUE: Prédictions contiennent des Inf!")

if preds_loaded.min() < 0.0 or preds_loaded.max() > 1.0:
    issues.append("⚠️  WARNING: Prédictions hors de [0, 1] (problème d'activation?)")

# Vérifier que le modèle chargé donne les mêmes prédictions
preds_original = model.predict(X_test, verbose=0)
if not np.allclose(preds_original, preds_loaded, atol=1e-5):
    issues.append("⚠️  WARNING: Prédictions différentes entre modèle original et chargé!")

if issues:
    print("\n🔴 PROBLÈMES DÉTECTÉS:")
    for issue in issues:
        print(f"   {issue}")
    print("\n   Le problème vient probablement de:")
    print("   1. Architecture du modèle")
    print("   2. Version de TensorFlow")
    print("   3. Problème de sauvegarde/chargement")
    success = False
else:
    print("\n✅ TOUS LES TESTS PASSENT!")
    print("   Le modèle peut être entraîné, sauvegardé et chargé correctement.")
    print("   Si l'évaluation échoue, le problème vient probablement:")
    print("   1. Du fichier de modèle spécifique que vous utilisez")
    print("   2. De l'entraînement qui n'a pas fonctionné")
    print("   3. D'un fichier corrompu")
    success = True

# Nettoyer
print("\n🧹 Nettoyage...")
if os.path.exists(test_model_path):
    os.remove(test_model_path)
    print(f"   Fichier temporaire supprimé: {test_model_path}")

print("\n" + "=" * 70)
if success:
    print("✅ TEST TERMINÉ AVEC SUCCÈS")
else:
    print("❌ TEST ÉCHOUÉ - Voir les problèmes ci-dessus")
print("=" * 70)

