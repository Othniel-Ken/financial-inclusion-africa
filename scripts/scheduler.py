import subprocess
import sys
import time
from pathlib import Path

import schedule

ROOT = Path(__file__).resolve().parent.parent
RETRAIN = ROOT / "scripts" / "retrain.py"


def job():
    print("Lancement du réentraînement...", flush=True)
    subprocess.run([sys.executable, str(RETRAIN)], check=False)


# Pour tester : toutes les minutes
schedule.every().day.at("02:00").do(job)
# Pour le rendu : une fois par jour à 2h du matin
# schedule.every().day.at("02:00").do(job)

print("Planificateur démarré.", flush=True)
while True:
    schedule.run_pending()
    time.sleep(10)
    