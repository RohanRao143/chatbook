        mkdir my-fastapi-app
        cd my-fastapi-app
        uv init .


        uv add -r ./requirement.txt

        uv run uvicorn main:app --reload

