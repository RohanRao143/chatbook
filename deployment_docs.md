        mkdir my-fastapi-app
        cd my-fastapi-app
        uv init .


        uv add -r ./requirement.txt

        uv run uvicorn app:app --reload

        gunicorn app:app --workers 4 --worker-class 



Use renderer or railway