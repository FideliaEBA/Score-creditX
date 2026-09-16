"""Génère un jeu de données SIMULÉ de demandes de crédit en microfinance.

Les données sont fictives (elles imitent des profils d'emprunteurs) : elles servent
à apprendre et à démontrer la méthode, pas à prendre de vraies décisions.
Utilisation : python generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

Path("data").mkdir(exist_ok=True)
rng = np.random.default_rng(42)
N = 2000

age = np.clip(rng.normal(36, 10, N), 20, 65).round()
revenu = (np.exp(rng.normal(11.6, 0.5, N)) / 5000).round() * 5000          # FCFA / mois
anciennete = np.clip(rng.exponential(5, N), 0, 30).round(1)                  # années d'activité
duree = rng.choice([6, 12, 18, 24], N)
montant = np.clip((revenu * rng.uniform(0.5, 6, N) / 5000).round() * 5000, 20000, 2_000_000)
nb_credits = np.clip(rng.poisson(1.5, N), 0, 8)
retards = rng.binomial(nb_credits, 0.2)
epargne = (revenu * rng.uniform(0, 3, N) * rng.binomial(1, 0.7, N) / 5000).round() * 5000
garant = rng.binomial(1, 0.45, N)

ratio_mens = (montant / duree) / revenu
ratio_ep = np.clip(epargne / revenu, 0, 3)

# Probabilité de défaut : plus la mensualité pèse sur le revenu et plus il y a de retards, plus le risque monte
z = (-2.7 + 4.2 * ratio_mens + 0.6 * retards - 0.07 * anciennete
     - 0.02 * (age - 36) - 0.35 * ratio_ep - 0.7 * garant + rng.normal(0, 0.5, N))
defaut = rng.binomial(1, 1 / (1 + np.exp(-z)))

df = pd.DataFrame({
    "age": age.astype(int), "revenu_mensuel": revenu.astype(int),
    "anciennete_activite": anciennete, "montant_demande": montant.astype(int),
    "duree_mois": duree, "nb_credits_precedents": nb_credits,
    "retards_precedents": retards, "epargne": epargne.astype(int),
    "a_garant": garant, "defaut": defaut,
})
df.to_csv("data/clients_microfinance.csv", index=False)
print(f"{N} clients générés — taux de défaut : {defaut.mean():.1%}")
