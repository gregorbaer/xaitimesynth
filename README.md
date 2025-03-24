# ds-project-template

## Setting up the project

Requirements: Have `uv`, `pyenv`, and `direnv` installed.

To set up the project as a whole with python environment and initial package installation in one command, run:

```bash
make setup
```

This will install the required python version if necessary and a list of default packages.
To change the default python version and packages, adjust the `config.mk` file.

Otherwise, or if it does not work, follow the steps below.

### Manual steps

1. **Create a new Python version with pyenv**: Download the necessary python version if it is not already installed on your system. Let’s say you want to install Python 3.12. You can do so with the following command:

    ```bash
    pyenv install 3.12
    ```
    - Set the local python environment with `pyenv local 3.12` to use a specific python version for your project. `uv` should use your local python version to figure out which python to use for installing packages.

2. **Create a new virtual environment with uv**: Navigate to your project root directory and create a new virtual environment using `uv`:

    ```bash
	uv venv
    ```

3. **Activate the environment automatically with direnv**: Create a `.envrc` file in your project directory with the following content:

    ```bash
    echo "source .venv/bin/activate" > .envrc
    ```

    Then allow it with `direnv`:

    ```bash
    direnv allow .
    ```

4. **Install necessary packages**: Now, whenever you enter your project directory, the virtual environment `myenv` will be activated automatically. You can packages with the following command:

    ```bash
    uv pip install pandas numpy
    ```

5. **Generate a "lock file" with dependences:** Use UV to generate a `requirements.txt` with all the dependencies and their versions.

    ```bash
    uv pip install pandas numpyuv pip freeze | uv pip compile - -o requirements.txt
    ```


## Useful UV commands

For commonds, see [UV documentation](https://github.com/astral-sh/uv).

Install a package:

```bash
uv pip install flask # Install Flask.
```

Generate new "lock file" (poetry equivalent), `requirements.txt` from installed dependencies

```bash
uv pip freeze | uv pip compile - -o requirements.txt  # Lock the current environment.
```

To sync a set of locked dependencies with the virtual environment:
This also uninstalls installed packages if they are not in `requirements.txt`.

```bash
uv pip sync requirements.txt  # Install from a requirements.txt file.
```

# Clean commiting

To rid all notebooks of their output before committing, you can use `nbstripout`. On bash/linux/WSL, use the following command:

```
find . -name '*.ipynb' -exec nbstripout --strip {} +
```

You can also use `nbdime` to properly be able to see the git diffs with notebook outputs.