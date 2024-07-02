# Contribution

Please install [VSCode](https://code.visualstudio.com/) and its extensions:

- Black Formatter
- isort
- Python
- Pylance
- Even Better TOML
- Prettier

Please install [Python](https://www.python.org/downloads/).

```bash
sudo apt install python3 python3-pip python3-venv
```

Please install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) with the official installer.

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.in-project true
```

To create your Python environment and install dependencies:

```bash
poetry install
```

To update dependencies:

```bash
poetry update
```

To clear poetry cache:

```bash
poetry cache clear --all .
```

To start:

```bash
poetry run python main.py
```
