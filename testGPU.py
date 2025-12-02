#!/usr/bin/env python3
"""
Script de test pour vérifier l'utilisation GPU sur le cluster ensicompute
"""

import os
import sys
import tensorflow as tf
import numpy as np
import datetime


def print_system_info():
    """Affiche les informations système"""
    print("=" * 60)
    print("INFORMATIONS SYSTEME")
    print("=" * 60)

    # Informations Python/TensorFlow
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")

    # Informations GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nNombre de GPUs physiques détectés: {len(gpus)}")

    for i, gpu in enumerate(gpus):
        print(f"\nGPU {i}:")
        print(f"  Nom: {gpu.name}")
        print(f"  Type de device: {gpu.device_type}")

    # Informations logiques (après distribution)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"\nNombre de GPUs logiques disponibles: {len(logical_gpus)}")

    # Variables d'environnement
    print("\nVariables d'environnement importantes:")
    print(f"  SLURM_GPUS: {os.environ.get('SLURM_GPUS', 'Non défini')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Non défini')}")
    print(f"  SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'Non en cours de job Slurm')}")


def test_gpu_operations():
    """Teste des opérations sur GPU"""
    print("\n" + "=" * 60)
    print("TESTS D'OPERATIONS GPU")
    print("=" * 60)

    # Test 1: Vérification basique
    try:
        if tf.test.is_gpu_available():
            print("✓ GPU disponible pour TensorFlow")
        else:
            print("✗ GPU non disponible pour TensorFlow")
    except Exception as e:
        print(f"Erreur lors de la vérification GPU: {e}")

    # Test 2: Placement automatique sur GPU
    print("\nTest de placement automatique:")
    with tf.device('/GPU:0'):
        try:
            # Création d'un tenseur sur GPU
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

            # Opération matricielle
            c = tf.matmul(a, b)

            print(f"✓ Tenseur a créé sur: {a.device}")
            print(f"✓ Tenseur b créé sur: {b.device}")
            print(f"✓ Résultat c sur: {c.device}")
            print(f"  Valeur de c:\n{c.numpy()}")

        except Exception as e:
            print(f"✗ Erreur lors du placement sur GPU: {e}")

    # Test 3: Comparaison performance CPU/GPU
    print("\nTest de performance (multiplication matricielle):")

    # Création de grandes matrices
    size = 5000
    print(f"  Taille des matrices: {size}x{size}")

    # Sur CPU
    start_time = datetime.datetime.now()
    with tf.device('/CPU:0'):
        cpu_a = tf.random.normal([size, size])
        cpu_b = tf.random.normal([size, size])
        cpu_c = tf.matmul(cpu_a, cpu_b)
        _ = cpu_c.numpy()  # Force l'exécution
    cpu_time = (datetime.datetime.now() - start_time).total_seconds()
    print(f"  Temps CPU: {cpu_time:.2f} secondes")

    # Sur GPU (si disponible)
    if len(tf.config.list_physical_devices('GPU')) > 0:
        start_time = datetime.datetime.now()
        with tf.device('/GPU:0'):
            gpu_a = tf.random.normal([size, size])
            gpu_b = tf.random.normal([size, size])
            gpu_c = tf.matmul(gpu_a, gpu_b)
            _ = gpu_c.numpy()  # Force l'exécution
        gpu_time = (datetime.datetime.now() - start_time).total_seconds()
        print(f"  Temps GPU: {gpu_time:.2f} secondes")

        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"  Accélération GPU: {speedup:.2f}x")
    else:
        print("  Test GPU skipé (pas de GPU disponible)")


def test_memory_gpu():
    """Test l'allocation mémoire GPU"""
    print("\n" + "=" * 60)
    print("INFORMATIONS MEMOIRE GPU")
    print("=" * 60)

    if len(tf.config.list_physical_devices('GPU')) > 0:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for i, gpu in enumerate(gpus):
                # Cette fonction nécessite parfois des permissions spécifiques
                print(f"\nGPU {i}:")

                # Mémoire GPU via TF
                details = tf.config.experimental.get_device_details(gpu)
                if details:
                    print(f"  Détails: {details}")

                # Tentative de récupération de l'utilisation mémoire
                from tensorflow.python.client import device_lib
                local_device_protos = device_lib.list_local_devices()
                for device in local_device_protos:
                    if 'GPU' in device.name:
                        print(f"  Nom complet: {device.name}")
                        print(f"  Mémoire totale: {device.memory_limit / 1e9:.2f} GB")
        except Exception as e:
            print(f"  Impossible d'obtenir les détails GPU: {e}")
    else:
        print("Aucun GPU détecté pour les tests mémoire")


def main():
    """Fonction principale"""
    print("SCRIPT DE TEST GPU POUR CLUSTER ENSICOMPUTE")
    print("Date d'exécution:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Vérifier la mémoire disponible
    print("\nMémoire système disponible:")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"  Total: {memory.total / 1e9:.2f} GB")
        print(f"  Disponible: {memory.available / 1e9:.2f} GB")
        print(f"  Utilisé: {memory.percent}%")
    except ImportError:
        print("  psutil non installé, installation avec: pip install psutil")

    # Exécuter les tests
    print_system_info()
    test_gpu_operations()
    test_memory_gpu()

    # Résumé final
    print("\n" + "=" * 60)
    print("RESUME FINAL")
    print("=" * 60)

    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        print(f"✅ SUCCES: {len(gpus)} GPU(s) détecté(s)")
        print("   TensorFlow peut utiliser le GPU sur ce cluster.")
        print("   N'oubliez pas d'utiliser '--gres=shard:1' avec srun/sbatch")
    else:
        print(f"❌ ATTENTION: Aucun GPU détecté")
        print("   Raisons possibles:")
        print("   1. Le job n'a pas été soumis avec --gres=shard:1")
        print("   2. TensorFlow n'est pas installé avec le support CUDA")
        print("   3. Les drivers GPU ne sont pas accessibles")
        print("\n   Pour exécuter sur GPU, utilisez:")
        print("   srun --gres=shard:1 --cpus-per-task=4 --mem=8GB python test_gpu.py")


if __name__ == "__main__":
    main()