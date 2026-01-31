# SoTa-Cross-Encoders


## Installation

To install this repository, first ensure you have `git` and `uv` installed.

1.  Clone the repository and its submodules:
    ```bash
    git clone --recurse-submodules git@git.isir.upmc.fr:morand/sota-cross-encoders.git
    cd sota-cross-encoders
    ```
    If you have already cloned the repository without `--recurse-submodules`, you can initialize and update them with:
    ```bash
    git submodule update --init --recursive
    ```

2.  Synchronize the Python dependencies using `uv`:
    ```bash
    uv sync
    ```
