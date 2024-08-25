## Contribution

### Installation

Please install [VSCode](https://code.visualstudio.com/) and its extensions:

- Auto Rename Tag
- ES7+ React/Redux/React-Native snippets
- Highlight Matching Tag
- ESLint
- Tailwind CSS IntelliSense
- Prettier - Code formatter
- Git Graph
- vscode-pdf
- One Dark Pro (optional)
- Material Icon Theme (optional)

Please install [git](https://git-scm.com/download/linux):

```bash
sudo apt install git
```

Please install [Node.js and npm](https://nodejs.org/en/download/package-manager).

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source $HOME/.bashrc
nvm install 20
node -v
npm -v
```

Please install the dependencies.

```bash
npm install
```

If all worked, congratulations! You are ready to contribute!

### Commands

To start the application in development mode.

```bash
npm run dev
```

To update the dependencies.

```bash
npm update --save
```

### Deployment

For Github Pages:

- Launch the following command

```bash
status=$(git status --porcelain)
if [[ -n $status ]]; then
    echo "Error: There are uncommitted changes in the repository."
    echo "$status"
else
    git worktree add gh-pages && cd gh-pages
    cd frontend && npm run build && cd ..
    mv frontend/out .
    ls -A | grep -Ev "^(\.git|out)$" | xargs rm -rf
    mv out/* . && rm -rf out
    touch .nojekyll
    git add .
    git commit -m "DONE: Add 'Deployment for Github Pages'"
    git push --set-upstream origin gh-pages --force
    git worktree remove gh-pages && cd .. && git branch -D gh-pages
fi
```

- (Only if GitHub free plan) From the repository: `General` > `Danger Zone` section, click-on `Change visibility` to make it public
- From the repository: `Settings` > `Pages` > `Branch` section, select `gh-pages` and `/ (root)`
