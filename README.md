# S3 Model Offloader

Web UI locale pour offloader tes modèles vers S3 et les restaurer en 1 clic.

## Setup

```bash
cd s3_offloader
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set MODELS_ROOT, S3_BUCKET

python app.py
# → http://localhost:5050
```

## Fonctionnement

- **Local Files tab** : browse ton dossier de modèles, coche les fichiers, "Send to S3"
- **On S3 tab** : liste tout ce qui est sur S3, coche, "Restore to Original Location" — aucune saisie requise
- **Config tab** : change le root dir, bucket, prefix à la volée

## Comment la restauration sait où aller ?

Le S3 key est construit comme `{prefix}{relative_path}` depuis `MODELS_ROOT`.  
Exemple : `/home/mehdi/models/checkpoints/v1.safetensors` → S3 key : `models-offload/checkpoints/v1.safetensors`  
À la restauration : `s3_key → strip prefix → rejoin MODELS_ROOT → recréer les dossiers → download`.

Aucun fichier de metadata externe, tout est dans la structure du S3 key.

## Extensions scannées

`.safetensors`, `.ckpt`, `.pt`, `.pth`, `.bin`, `.gguf`, `.ggml`, `.pkl`

## AWS credentials

Utilise boto3 standard : `~/.aws/credentials`, variables d'env, ou IAM role.  
Ou `AWS_PROFILE` dans `.env` pour un profil nommé.
