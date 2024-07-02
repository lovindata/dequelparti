# Contribution

Please install [Node.js and npm](https://nodejs.org/en/download/package-manager).

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source $HOME/.bashrc
nvm install 20
node -v
npm -v
```

Please install VSCode extensions.

- Auto Rename Tag
- ES7+ React/Redux/React-Native snippets
- Highlight Matching Tag
- ESLint
- Prettier - Code formatter
- Tailwind CSS IntelliSense

Please install the dependencies.

```bash
npm install
```

To update the dependencies.

```bash
npm update --save
```

To start the application in development mode.

```bash
npm run dev
```

# Deployment

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
