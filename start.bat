start cmd /k "D: && .\CEnv\Scripts\activate && python -m uvicorn app.main:app --reload"


start cmd /k "D: && .\CEnv\Scripts\activate && python .\app\manage_cache.py"
