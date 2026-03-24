import requests
import time
import subprocess
import os
import signal

# Start the uvicorn server
env = os.environ.copy()
env['API_KEY'] = 'test1234'
proc = subprocess.Popen(["python", "-m", "uvicorn", "api.app:app", "--port", "8000"], env=env)

# Wait for server to start
time.sleep(3)

print("Testing /health ...")
try:
    health = requests.get("http://localhost:8000/health")
    print(health.json())
except Exception as e:
    print(f"Health check failed: {e}")

print("Testing /predict ...")
payload = {
    "player_id": "P9999",
    "Age": 25.0,
    "Gender": "Male",
    "Location": "North America",
    "GameGenre": "RPG",
    "PlayTimeHours": 10.5,
    "InGamePurchases": 1,
    "GameDifficulty": "Hard",
    "SessionsPerWeek": 5.0,
    "AvgSessionDurationMinutes": 60.0,
    "PlayerLevel": 20.0,
    "AchievementsUnlocked": 15.0,
    "TotalWeeklyMinutes": 300.0,
    "AchievementsPerLevel": 0.75
}

headers = {"X-API-Key": "test1234"}
try:
    resp = requests.post("http://localhost:8000/predict", json=payload, headers=headers)
    print(f"Status: {resp.status_code}")
    print(resp.json())
except Exception as e:
    print(f"Prediction failed: {e}")

# Kill the server
proc.send_signal(signal.CTRL_C_EVENT)
proc.wait(timeout=3)
