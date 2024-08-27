# DQChat

[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)

> [!NOTE]
> We are making huge refactoring to package structure. Please refer to [GitHub Issues](https://github.com/junnoslab/DQChat-langGraph/issues) for progress.

## Flow Chart

![Flow Chart](graph.png)

## How to start

We use [`pixi`](https://pixi.sh/latest/) for package management tool.

```shell
# With official shell script
curl -fsSL https://pixi.sh/install.sh | bash

# Simply using Homebrew
brew install pixi
```

```shell
pixi shell # enables venv

pixi r r # run inference mode
pixi r gd # run RAFT dataset generator
pixi r t # run RAFT fine-tuning
```
