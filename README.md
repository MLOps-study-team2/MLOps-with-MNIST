# MLOps with MNIST

이 프로젝트는 MNIST 데이터셋을 사용한 간단한 이미지 분류 모델을 구축하는 예제입니다. `Poetry`를 사용하여 파이썬 환경을 관리하며, `model_train.py` 스크립트를 실행하여 모델을 학습하고 평가할 수 있습니다.

## 요구 사항

- Python 3.7 이상
- Poetry

## Poetry 설치

### Windows

1. Windows 터미널(명령 프롬프트, PowerShell 등)을 엽니다.
2. 다음 명령어를 입력하여 Poetry를 설치합니다:

    ```bash
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    ```

3. 설치가 완료되면, 다음 명령어를 사용하여 Poetry를 PATH에 추가합니다:

    ```bash
    $Env:Path += ";$Env:APPDATA\Python\Scripts"
    ```

### macOS

1. 터미널을 엽니다.
2. 다음 명령어를 입력하여 Poetry를 설치합니다:

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    brew로 설치하는 경우 다음을 입력합니다. 
    ```bash
    brew install poetry
    ```


3. 설치가 완료되면, 다음 명령어를 사용하여 Poetry를 PATH에 추가합니다:

    ```bash
    export PATH="$HOME/.local/bin:$PATH"
    ```

4. 위의 명령어를 `.bashrc`, `.zshrc` 또는 다른 셸 초기화 파일에 추가하여 터미널을 열 때마다 PATH가 설정되도록 합니다.

## 프로젝트 설정

1. 프로젝트 디렉토리로 이동합니다:

    ```bash
    cd /path/to/your/project
    ```

2. Poetry 환경을 초기화하고 종속성을 설치합니다:

    ```bash
    poetry install
    ```

    이 명령어는 `pyproject.toml` 파일에 정의된 모든 의존성을 설치하고 가상 환경을 구성합니다.

## 모델 학습 및 평가

`model_train.py` 스크립트를 실행하여 모델을 학습하고 평가할 수 있습니다.

```bash
poetry run python -u "C:\path\to\your\project\model_train.py"
```

### macOS

```bash
poetry run python -u "/path/to/your/project/model_train.py"
```

위 명령어는 구성된 `Poetry` 환경에서 Python 스크립트를 실행합니다. 학습이 완료되면 최종 정확도가 출력됩니다.

## 문제 해결

설치나 실행 중 문제가 발생하면 다음을 확인해보세요:

- Python 버전이 3.7 이상인지 확인하세요.
- Poetry가 올바르게 설치되었는지, 그리고 PATH에 추가되었는지 확인하세요.
- 위의 명령어를 정확히 입력했는지 확인하세요.