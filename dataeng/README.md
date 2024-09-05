## Contribution

### Installation

Please install [VSCode](https://code.visualstudio.com/) and its extensions:

- Python
- Pylance
- Black Formatter
- isort
- Even Better TOML
- Prettier - Code formatter
- Git Graph
- vscode-pdf
- One Dark Pro (optional)
- Material Icon Theme (optional)

Please install [git](https://git-scm.com/download/linux):

```bash
sudo apt install git
```

Please install [Python](https://www.python.org/downloads/).

```bash
sudo apt install python3 python3-pip python3-venv
```

Please install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer):

```bash
curl -sSL https://install.python-poetry.org | python3 -
if ! command -v poetry &> /dev/null; then
  echo '\n# Poetry\nexport PATH=$HOME/.local/bin:$PATH' >> $HOME/.bashrc
  source $HOME/.bashrc
else
  echo "Poetry already added to PATH."
fi
poetry config virtualenvs.in-project true
```

To create your Python environment and install dependencies:

```bash
poetry install
```

Please reload the VSCode window:

- In VSCode, press `CTRL + SHIFT + P`
- Click on `Developer: Reload Window`

If all worked, congratulations! You are ready to contribute!

### Commands

To launch:

```bash
poetry run python main.py
```

To launch with the debugger:

- Open `main.py`
- Click on selection menu close to the ▶️ icon
- Click on `Python Debugger: Debug Python File`

To update dependencies:

```bash
poetry update
```

### Other commands

To clear poetry cache:

```bash
poetry cache clear --all .
```
