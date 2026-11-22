# Projet Image : Anonymisation des visages

## Objectifs

Développer un model d'anonymisation des visages réversible composé de :
- D'un auto-encodeur
- D'un discriminateur
- D'un encryptor/decryptor de vecteurs latent


## Lancer l'entrainement

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 train.py
```

## Localisation des compte-rendus hebdomadaires

`~/documents/reports/pdf/report_<weekID>`
