#!/usr/bin/env python3
"""
Analyze feature importance and reduction impact
Usage: python feature_analysis.py
"""

import pandas as pd
import numpy as np

def analyze_features():
    # Load feature summary
    feature_summary = pd.read_csv('selected_users_dataset/results/feature_summary.csv')

    print("=== FEATURE IMPORTANCE ANALYSIS ===")

    # Overall statistics
    total_features = len(feature_summary)
    features_1pct = len(feature_summary[feature_summary['mean_importance'] >= 1.0])
    features_5pct = len(feature_summary[feature_summary['mean_importance'] >= 5.0])

    print(f"Total features: {total_features}")
    print(f"Features ≥1% importance: {features_1pct} ({features_1pct/total_features*100:.1f}%)")
    print(f"Features ≥5% importance: {features_5pct} ({features_5pct/total_features*100:.1f}%)")

    # Top features
    print(f"\nTop 10 most important features:")
    top_features = feature_summary.head(10)
    for _, row in top_features.iterrows():
        print(f"{row['feature']}: {row['mean_importance']:.2f}% (±{row['std_importance']:.2f}%)")

    # Universal features (important for all users)
    universal_features = feature_summary[feature_summary['user_count'] == 13]
    print(f"\nUniversal features (important for all 13 users): {len(universal_features)}")

    # Coverage analysis
    cumulative_importance = feature_summary['mean_importance'].cumsum()
    features_for_90pct = len(cumulative_importance[cumulative_importance <= 90])
    features_for_95pct = len(cumulative_importance[cumulative_importance <= 95])

    print(f"\nCoverage analysis:")
    print(f"Features needed for 90% importance: {features_for_90pct}")
    print(f"Features needed for 95% importance: {features_for_95pct}")

    # Individual user analysis
    print(f"\n=== PER-USER ANALYSIS ===")

    user_stats = []
    for user_file in ['P124', 'P135', 'P045', 'P050', 'P052', 'P071', 'P133', 'P046', 'P056', 'P048', 'P024', 'P094', 'P105']:
        try:
            user_features = pd.read_csv(f'selected_users_dataset/results/{user_file}_feature_importance.csv')
            features_1pct = len(user_features[user_features['importance'] >= 1.0])
            importance_sum = user_features[user_features['importance'] >= 1.0]['importance'].sum()

            user_stats.append({
                'user': user_file,
                'features_1pct': features_1pct,
                'importance_retained': importance_sum,
                'reduction_pct': (216 - features_1pct) / 216 * 100
            })

        except FileNotFoundError:
            continue

    user_stats_df = pd.DataFrame(user_stats)
    print(f"Average features ≥1% per user: {user_stats_df['features_1pct'].mean():.1f}")
    print(f"Average importance retained: {user_stats_df['importance_retained'].mean():.1f}%")
    print(f"Average feature reduction: {user_stats_df['reduction_pct'].mean():.1f}%")

    return feature_summary, user_stats_df

if __name__ == "__main__":
    analyze_features()