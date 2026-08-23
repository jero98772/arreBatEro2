
uv run python -c "import json; from main import app; json.dump(app.openapi(), open('docs/source/openapi.json', 'w'))"
uv run sphinx-build -b html -E docs/source docs/_build