"""Entraîne un modèle de score de risque de crédit et exporte ses paramètres pour l'interface web.

Utilisation : python train_model.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "anciennete_activite", "ratio_mensualite_revenu", "ratio_epargne_revenu",
            "nb_credits_precedents", "retards_precedents", "a_garant"]


def preparer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ratio_mensualite_revenu"] = (df["montant_demande"] / df["duree_mois"]) / df["revenu_mensuel"]
    df["ratio_epargne_revenu"] = (df["epargne"] / df["revenu_mensuel"]).clip(0, 3)
    return df


def main() -> None:
    Path("results").mkdir(exist_ok=True)
    df = preparer(pd.read_csv("data/clients_microfinance.csv"))
    X, y = df[FEATURES], df["defaut"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_tr)
    logit = LogisticRegression(max_iter=1000).fit(scaler.transform(X_tr), y_tr)
    foret = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42).fit(X_tr, y_tr)

    p_logit = logit.predict_proba(scaler.transform(X_te))[:, 1]
    p_foret = foret.predict_proba(X_te)[:, 1]
    lignes = [f"Taux de défaut : {y.mean():.1%} ({len(df)} clients, {len(X_te)} en test)", ""]
    for nom, p in [("Régression logistique", p_logit), ("Forêt aléatoire", p_foret)]:
        pred = (p >= 0.5).astype(int)
        lignes += [f"{nom} : précision {accuracy_score(y_te, pred):.3f} | AUC {roc_auc_score(y_te, p):.3f}",
                   f"  matrice de confusion [[VN, FP], [FN, VP]] : {confusion_matrix(y_te, pred).tolist()}"]
    rapport = "\n".join(lignes)
    print(rapport)
    open("results/metriques.txt", "w", encoding="utf-8").write(rapport + "\n")

    # Courbes ROC
    plt.figure(figsize=(6, 5))
    for nom, p in [("Régression logistique", p_logit), ("Forêt aléatoire", p_foret)]:
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, label=f"{nom} (AUC {roc_auc_score(y_te, p):.2f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("Faux positifs"); plt.ylabel("Vrais positifs"); plt.title("Courbes ROC")
    plt.legend(); plt.tight_layout(); plt.savefig("results/roc.png", dpi=150); plt.close()

    # Influence des variables (coefficients standardisés)
    coefs = pd.Series(logit.coef_[0], index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5))
    plt.barh(coefs.index, coefs.values, color=["#2f9e6b" if v < 0 else "#d9534f" for v in coefs.values])
    plt.axvline(0, color="grey"); plt.xlabel("Effet sur le risque de défaut (rouge : augmente, vert : diminue)")
    plt.title("Influence des variables"); plt.tight_layout(); plt.savefig("results/influence_variables.png", dpi=150); plt.close()

    # Export pour la page web (le modèle logistique est simple à recalculer en JavaScript)
    modele = {"features": FEATURES, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
              "coef": logit.coef_[0].tolist(), "intercept": float(logit.intercept_[0])}
    open("model.js", "w", encoding="utf-8").write("const MODEL = " + json.dumps(modele, indent=2) + ";\n")
    print("\nModèle exporté dans model.js")


if __name__ == "__main__":
    main()
