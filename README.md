This repository contains a Python package developed for creating Artificial Neural Networks through Decoupled Neural Interfaces.

### Project directory structure
The directory structure of this repository viewed from the top-level directory is:
```
.
├── src       # Python source code
└── tests     # Unit tests
```

### Setup virtual environment with `uv` package manager

1. **Install `uv` and download python:**
    Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
    With `uv` install, download python 3.12:
    ```
    uv python install 3.12
    ```

2. **Clone repository and navigate to project directory:**
    ```
    cd <path_to_dni_modules>
    ```

3. **Install dependencies (required for running tests):**
    ```
    uv sync
    ```

5. **Activate virtual environment:**
    ```
    . .venv/bin/activate
    ```

### Testing
To run the unit tests placed under the `tests` directory:
``` 
cd <path_to_dni_modules>
pytest
```
