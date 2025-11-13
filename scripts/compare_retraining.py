#!/usr/bin/env python3
"""
Compare results before and after retraining.
Shows improvement metrics for Hard and Medium modes.
"""

import json
from pathlib import Path

def load_results(file_path):
    """Load evaluation results from JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)

def print_comparison():
    """Print comparison of old vs new results."""

    print("=" * 90)
    print("📊 COMPARACIÓN DE RESULTADOS - ANTES VS DESPUÉS DEL REENTRENAMIENTO")
    print("=" * 90)
    print()

    # Check which files exist
    hard_old = Path("results/hard_evaluation.json")
    hard_new = Path("results/hard_evaluation_4M.json")
    medium_old = Path("results/medium_evaluation.json")
    medium_new = Path("results/medium_evaluation_3M.json")

    # Hard Mode comparison
    if hard_old.exists() and hard_new.exists():
        print("🔴 HARD MODE - Reentrenamiento 2M → 4M steps")
        print("-" * 90)

        old_data = load_results(hard_old)
        new_data = load_results(hard_new)

        old_eff = old_data['statistics']['mean_efficiency']
        new_eff = new_data['statistics']['mean_efficiency']
        old_std = old_data['statistics']['std']
        new_std = new_data['statistics']['std']
        old_gini = old_data['statistics'].get('mean_gini', 0)
        new_gini = new_data['statistics'].get('mean_gini', 0)

        old_excellent = old_data['performance_tiers'].get('excellent_70plus',
                                                          old_data['performance_tiers'].get('excellent', 0))
        new_excellent = new_data['performance_tiers'].get('excellent_70plus',
                                                          new_data['performance_tiers'].get('excellent', 0))
        old_below40 = old_data['performance_tiers']['below_40']
        new_below40 = new_data['performance_tiers']['below_40']

        diff_eff = new_eff - old_eff
        diff_std = new_std - old_std
        diff_gini = new_gini - old_gini
        diff_excellent = new_excellent - old_excellent
        diff_below40 = new_below40 - old_below40

        print(f"  Eficiencia:       {old_eff:6.2f}% → {new_eff:6.2f}%   ({diff_eff:+.2f}pp) {'✅' if diff_eff > 2 else '⚠️'}")
        print(f"  Std Deviation:    {old_std:6.1f} → {new_std:6.1f}     ({diff_std:+.1f})   {'✅' if diff_std < -10 else '⚠️'}")
        print(f"  Gini Coefficient: {old_gini:6.3f} → {new_gini:6.3f}   ({diff_gini:+.3f})  {'✅' if diff_gini < -0.02 else '⚠️'}")
        print(f"  Episodios ≥70%:   {old_excellent:3d}/100 → {new_excellent:3d}/100   ({diff_excellent:+d})       {'✅' if diff_excellent > 3 else '⚠️'}")
        print(f"  Episodios <40%:   {old_below40:3d}/100 → {new_below40:3d}/100   ({diff_below40:+d})       {'✅' if diff_below40 < -5 else '⚠️'}")
        print()

        # Overall assessment
        improvements = sum([
            diff_eff > 2,
            diff_std < -10,
            diff_gini < -0.02,
            diff_excellent > 3,
            diff_below40 < -5
        ])

        if improvements >= 4:
            print("  🎉 MEJORA SIGNIFICATIVA - Todos los objetivos alcanzados")
        elif improvements >= 3:
            print("  ✅ MEJORA NOTABLE - La mayoría de objetivos alcanzados")
        elif improvements >= 2:
            print("  ⚠️  MEJORA MODERADA - Algunos objetivos alcanzados")
        else:
            print("  ❌ MEJORA INSUFICIENTE - Considerar más timesteps o ajustes")
        print()

    # Medium Mode comparison
    if medium_old.exists() and medium_new.exists():
        print("🟡 MEDIUM MODE - Reentrenamiento 2M → 3M steps")
        print("-" * 90)

        old_data = load_results(medium_old)
        new_data = load_results(medium_new)

        old_eff = old_data['statistics']['mean_efficiency']
        new_eff = new_data['statistics']['mean_efficiency']
        old_std = old_data['statistics']['std']
        new_std = new_data['statistics']['std']
        old_gini = old_data['statistics'].get('mean_gini', 0)
        new_gini = new_data['statistics'].get('mean_gini', 0)

        old_excellent = old_data['performance_tiers'].get('excellent_70plus',
                                                          old_data['performance_tiers'].get('excellent', 0))
        new_excellent = new_data['performance_tiers'].get('excellent_70plus',
                                                          new_data['performance_tiers'].get('excellent', 0))

        diff_eff = new_eff - old_eff
        diff_std = new_std - old_std
        diff_gini = new_gini - old_gini
        diff_excellent = new_excellent - old_excellent

        print(f"  Eficiencia:       {old_eff:6.2f}% → {new_eff:6.2f}%   ({diff_eff:+.2f}pp) {'✅' if diff_eff > 2 else '⚠️'}")
        print(f"  Std Deviation:    {old_std:6.1f} → {new_std:6.1f}     ({diff_std:+.1f})   {'✅' if diff_std < -20 else '⚠️'}")
        print(f"  Gini Coefficient: {old_gini:6.3f} → {new_gini:6.3f}   ({diff_gini:+.3f})  {'✅' if diff_gini < -0.01 else '⚠️'}")
        print(f"  Episodios ≥70%:   {old_excellent:3d}/100 → {new_excellent:3d}/100   ({diff_excellent:+d})       {'✅' if diff_excellent > 10 else '⚠️'}")
        print()

        # Overall assessment
        improvements = sum([
            diff_eff > 2,
            diff_std < -20,
            diff_gini < -0.01,
            diff_excellent > 10
        ])

        if improvements >= 3:
            print("  🎉 MEJORA SIGNIFICATIVA - Objetivos alcanzados")
        elif improvements >= 2:
            print("  ✅ MEJORA NOTABLE - Resultados prometedores")
        else:
            print("  ⚠️  MEJORA MODERADA - Considerar ajustes adicionales")
        print()

    print("=" * 90)
    print()

    # Summary
    print("📈 RESUMEN FINAL")
    print("=" * 90)
    print()

    if hard_old.exists() and hard_new.exists() and medium_old.exists() and medium_new.exists():
        hard_data = load_results(hard_new)
        medium_data = load_results(medium_new)

        print("Progresión de eficiencia con nuevos modelos:")
        print(f"  Easy   → 87.22% (sin cambios)")
        print(f"  Medium → {medium_data['statistics']['mean_efficiency']:.2f}% (reentrenado 3M steps)")
        print(f"  Hard   → {hard_data['statistics']['mean_efficiency']:.2f}% (reentrenado 4M steps)")
        print(f"  Expert → 37.12% (sin cambios)")
        print()

        # Check if targets met
        hard_eff = hard_data['statistics']['mean_efficiency']
        medium_eff = medium_data['statistics']['mean_efficiency']

        hard_ok = 48 <= hard_eff <= 52
        medium_ok = medium_eff >= 75

        if hard_ok and medium_ok:
            print("✅ Todos los objetivos de reentrenamiento alcanzados")
            print("   Listo para publicación académica")
        elif hard_ok or medium_ok:
            print("⚠️  Algunos objetivos alcanzados, otros pueden mejorar")
            print("   Considerar reentrenamiento adicional si es necesario")
        else:
            print("❌ Objetivos no alcanzados")
            print("   Reentrenamiento con más timesteps recomendado")
    else:
        print("⚠️  No se encontraron todos los archivos de resultados")
        print("   Ejecuta el reentrenamiento primero:")
        print("   bash scripts/retrain_gpu.sh")

    print()
    print("=" * 90)

if __name__ == "__main__":
    print_comparison()
