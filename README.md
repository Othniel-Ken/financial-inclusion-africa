# Financial Inclusion in Africa 

**Projet de Data Science — Analyse et prédiction de l'inclusion financière en Afrique**

## 1. Présentation du projet

L'inclusion financière constitue un enjeu majeur du développement économique en Afrique. Malgré les progrès réalisés dans les services financiers numériques, une partie importante de la population reste exclue du système bancaire traditionnel.

Ce projet vise à analyser les facteurs associés à la détention d'un compte bancaire et à développer un modèle de Machine Learning capable de prédire la probabilité qu'un individu possède un compte bancaire.

L'étude porte sur quatre pays africains :

- 🇰🇪 Kenya
- 🇷🇼 Rwanda
- 🇹🇿 Tanzanie
- 🇺🇬 Ouganda

Les données proviennent d'enquêtes réalisées entre 2016 et 2018.

## 2. Objectifs

Les principaux objectifs sont :

- Explorer les caractéristiques socio-économiques des populations étudiées.
- Identifier les facteurs associés à la détention d'un compte bancaire.
- Préparer et transformer les données pour la modélisation.
- Comparer plusieurs algorithmes de Machine Learning.
- Développer une API permettant d'effectuer des prédictions.
- Mettre en place une architecture permettant le réentraînement du modèle.

## 3. Source des données

Le projet utilise le jeu de données **Financial Inclusion in Africa**, disponible sur la plateforme Zindi.

Source : [Financial Inclusion in Africa — Zindi](https://zindi.africa/competitions/financial-inclusion-in-africa)

La variable cible est `bank_account` :

- `1` : l'individu possède un compte bancaire.
- `0` : l'individu ne possède pas de compte bancaire.

**Important :** les fichiers de données ne sont pas inclus dans ce dépôt. Ils doivent être obtenus depuis la source officielle et placés localement dans le dossier `data/`. Il est recommandé de respecter les conditions de licence et de distribution de Zindi.

Pour le réentraînement, le fichier nettoyé attendu est :

```text
data/
└── train_clean.csv
```

## 4. Structure du projet

```text
financial-inclusion-africa/
│
├── app/
│   ├── main.py
│   └── static/
│       └── index.html
│
├── models/
│   ├── colonnes_entree.json
│   └── modele_inclusion_financiere.joblib
│
├── scripts/
│   ├── retrain.py
│   ├── scheduler.py
│   └── train_dummy.py
│
├── EDA_Inclusion_Financiere.ipynb
├── pretraitement_modelisation.ipynb
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

## 5. Méthodologie

Le travail est organisé en plusieurs étapes.

### 5.1. Analyse exploratoire des données (EDA)

Cette étape permet de comprendre les données et d'identifier les tendances principales :

- Analyse de la distribution des variables.
- Étude de la répartition des comptes bancaires.
- Comparaison entre les quatre pays.
- Analyse des relations entre les variables socio-économiques et la détention d'un compte bancaire.
- Identification des incohérences et préparation des données.

Notebook : `EDA_Inclusion_Financiere.ipynb`

### 5.2. Prétraitement et feature engineering

Les opérations comprennent notamment :

- Nettoyage des données.
- Traitement des variables catégorielles et numériques.
- Création de variables dérivées.
- Encodage des variables.
- Normalisation des variables numériques.
- Séparation des données en ensembles d'entraînement et de test.
- Prise en compte du déséquilibre des classes.

Notebook : `pretraitement_modelisation.ipynb`

### 5.3. Modélisation

Plusieurs approches ont été étudiées, notamment :

- Régression logistique.
- Random Forest.
- XGBoost.
- Différentes stratégies de gestion du déséquilibre des classes.

Le choix du modèle tient compte de plusieurs métriques, en particulier le rappel de la classe positive, le F1-score, l'Average Precision et la Balanced Accuracy.

## 6. Résultats de la modélisation

Sur l'ensemble de test utilisé pendant l'expérimentation, le modèle XGBoost pondéré a obtenu les résultats suivants :

| Métrique | Résultat |
|---|---:|
| Accuracy | 0,79 |
| Balanced Accuracy | 0,7802 |
| Precision — classe 1 | 0,38 |
| Recall — classe 1 | 0,77 |
| F1-score — classe 1 | 0,51 |
| ROC-AUC | 0,8660 |
| Average Precision | 0,5794 |

### Interprétation

Le modèle détecte environ 77 % des individus possédant effectivement un compte bancaire dans l'ensemble de test.

Le rappel de la classe positive est particulièrement important dans cette étude : il permet d'évaluer la capacité du modèle à identifier les individus bancarisés.

L'Accuracy doit toutefois être interprétée avec prudence, car la classe majoritaire représente environ 85,92 % des observations. La Balanced Accuracy et l'Average Precision apportent donc des informations complémentaires essentielles.

Ces résultats correspondent à une expérimentation sur un jeu de test et ne constituent pas une garantie de performance en situation réelle.

## 7. Application de prédiction

Une API développée avec **FastAPI** permet de soumettre les caractéristiques d'un individu et d'obtenir une prédiction.

Le fichier principal est :

```text
app/main.py
```

L'interface utilisateur est disponible dans :

```text
app/static/index.html
```

Le modèle entraîné et les colonnes attendues sont stockés dans :

```text
models/
├── modele_inclusion_financiere.joblib
└── colonnes_entree.json
```

### Endpoints disponibles

| Endpoint | Méthode | Description |
|---|---|---|
| `/` | GET | Accès à l'interface |
| `/health` | GET | Vérification de l'état de l'API |
| `/predict` | POST | Prédiction de l'inclusion financière |

### Lancement local

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Démarrer l'API :

```bash
uvicorn app.main:app --reload
```

L'application sera accessible à l'adresse :

```text
http://127.0.0.1:8000
```

La documentation interactive de l'API est accessible ici :

```text
http://127.0.0.1:8000/docs
```

## 8. Conteneurisation avec Docker

Le projet contient les fichiers nécessaires à la construction et à l'orchestration des services Docker :

- `Dockerfile`
- `docker-compose.yml`

Pour construire et démarrer les services :

```bash
docker compose up --build
```

Pour arrêter les services :

```bash
docker compose down
```

**Remarque :** l'exécution complète avec Docker doit être vérifiée dans l'environnement de déploiement avant de considérer cette étape comme validée.

## 9. Réentraînement du modèle

Le projet intègre des scripts destinés à la gestion du réentraînement :

| Fichier | Rôle |
|---|---|
| `scripts/retrain.py` | Entraînement et évaluation d'un nouveau modèle |
| `scripts/scheduler.py` | Planification du réentraînement |
| `scripts/train_dummy.py` | Génération de données synthétiques pour une démonstration |

Le script de réentraînement attend le fichier :

```text
data/train_clean.csv
```

Le planificateur est configuré pour lancer le travail quotidiennement à 02 h 00.

Le fonctionnement du processus complet, notamment la disponibilité des données, la compatibilité du modèle et le remplacement du modèle existant, doit être testé avant une utilisation opérationnelle.

**Attention :** `train_dummy.py` utilise des données synthétiques et ne doit pas être utilisé pour produire des prédictions réelles.

## 10. Technologies utilisées

- **Langage :** Python
- **Analyse des données :** Pandas, NumPy
- **Visualisation :** Matplotlib, Seaborn
- **Machine Learning :** Scikit-learn, XGBoost, Imbalanced-learn
- **Sérialisation :** Joblib
- **API :** FastAPI, Pydantic, Uvicorn
- **Conteneurisation :** Docker, Docker Compose
- **Planification :** Schedule
- **Collaboration et versionnement :** Git, GitHub

## 11. Limites et perspectives

Plusieurs améliorations peuvent être envisagées :

- Améliorer la précision des prédictions positives.
- Tester d'autres algorithmes et stratégies de gestion du déséquilibre.
- Étudier la robustesse du modèle sur de nouvelles données.
- Ajouter des mécanismes de suivi des performances et de détection de dérive des données.
- Améliorer la validation des données reçues par l'API.
- Mettre en place des tests automatisés.
- Déployer l'application sur une infrastructure accessible en ligne.

## 12. Équipe projet

Projet réalisé dans le cadre d'un travail académique en Data Science.

**Thème :** Financial Inclusion in Africa

Les contributions couvrent l'analyse exploratoire, le prétraitement, la modélisation, le développement de l'API, l'automatisation et la documentation.

---

*Financial Inclusion in Africa — Exploiter la Data Science pour mieux comprendre et prédire l'inclusion financière en Afrique.*
