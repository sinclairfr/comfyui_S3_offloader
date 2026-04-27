# S3 Model Offloader

Web UI locale pour offloader tes modèles vers S3 et les restaurer en 1 clic.

## Setup

```bash
cd comfyui_S3_offloader
pip install -r requirements.txt

python app.py
```

## Lancement (port)

Par défaut, le serveur Flask écoute sur le port **8900**.

- URL locale (défaut): `http://localhost:8900`
- URL réseau local (défaut): `http://<IP_DE_TA_MACHINE>:8900`

Tu peux changer le port avec l'argument CLI `--port`.

Exemples:

```bash
# Port par défaut
python app.py

# Port personnalisé
python app.py --port 5050
```

Avec `--port 5050`, l'URL devient `http://localhost:5050`.

## Configuration (JSON)

Les settings runtime sont persistés dans [`settings.json`](settings.json).

### Priorité de configuration (important)

L'application charge d'abord les variables de [`.env`](.env), puis fusionne avec [`settings.json`](settings.json).

En pratique :

- [`settings.json`](settings.json) = **source de vérité runtime** (modifiée par l'onglet Config)
- [`.env`](.env) = **valeurs de bootstrap** (au démarrage / si clé absente)

Pour éviter la confusion, recommandé :

- garder les credentials/valeurs actives dans [`settings.json`](settings.json)
- ne pas laisser `AWS_PROFILE=` vide dans [`.env`](.env)
- définir `AWS_PROFILE` seulement si tu utilises réellement un profil nommé AWS

Fichier par défaut :

```json
{
  "models_root": "~/models",
  "s3_bucket": "",
  "s3_prefix": "models-offload/",
  "aws_profile": null,
  "include_personal_stuff": false,
  "personal_paths": [
    "/workspace/ComfyUI/custom_nodes",
    "/workspace/ComfyUI/user",
    "/workspacecomfyui_S3_offloader",
    "/workspace/medo_start.sh"
  ]
}
```

Tu peux aussi changer le chemin du fichier via la variable d'environnement `CONFIG_FILE`.

Exemple:

```bash
CONFIG_FILE=/chemin/vers/mes-settings.json python app.py
```

## Fonctionnement

- **Local Files tab** : browse ton dossier de modèles, coche les fichiers, "Send to S3"
  - recherche live dans l’arborescence locale
  - sélection "All" appliquée sur les résultats visibles (filtrés)
  - checkbox dossier = sélection/désélection récursive de tous les fichiers du dossier
  - badge `S3` visible aussi sur les dossiers parents si au moins un fichier descendant est déjà sur S3
- **On S3 tab** : liste tout ce qui est sur S3, coche, "Restore to Original Location" — aucune saisie requise
  - recherche live (nom, chemin relatif, key S3, chemin local)
  - vue arborescence (Tree) + bascule Tree/List
- **Config tab** : change `models_root`, `s3_bucket`, `s3_prefix`, `aws_profile`, `include_personal_stuff`, `personal_paths` à la volée (et sauvegarde dans [`settings.json`](settings.json))
  - bouton `↻ Restart App` pour redémarrer l’application depuis l’UI

## UI — recherche & navigation

- **Local Files**
  - Champ `Search local files...`
  - Filtre l’arborescence en temps réel
  - Les dossiers correspondants sont auto-dépliés pendant la recherche
  - Le coche d’un dossier reflète correctement l’état "tout sélectionné" des fichiers descendants
  - Le badge `S3` est affiché sur les fichiers **et** dossiers contenant des fichiers uploadés

- **On S3**
  - Champ `Search path, filename, key...`
  - Filtre en temps réel
  - Boutons `🌲 Tree` / `☰ List` pour changer de vue

## Comment la restauration sait où aller ?

Le S3 key est construit à partir de `s3_prefix` + un chemin relatif à la source :

- modèles : `models/<relative_path>`
- chemins perso : `personal/<slug>/<relative_path>`

Exemple modèle : `/home/mehdi/models/checkpoints/v1.safetensors` → `models-offload/models/checkpoints/v1.safetensors`.

Aucun fichier de metadata externe, tout est dans la structure du S3 key.

## API utile (UI)

- `POST /api/restart` : redémarre le process Flask (appelé par le bouton `Restart App` dans l’onglet Config).

## Extensions scannées

`.safetensors`, `.ckpt`, `.pt`, `.pth`, `.bin`, `.gguf`, `.ggml`, `.pkl`

## AWS credentials

Utilise boto3 standard : `~/.aws/credentials`, variables d'env, ou IAM role.  
Ou `AWS_PROFILE` dans `.env` pour un profil nommé.

⚠️ Évite `AWS_PROFILE=` vide dans [`.env`](.env), cela peut provoquer l'erreur
`The config profile () could not be found`.
