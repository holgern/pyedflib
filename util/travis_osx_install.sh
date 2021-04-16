#!/bin/bash
brew update
brew install ccache

# install pyenv
git clone --depth 1 https://github.com/pyenv/pyenv ~/.pyenv
PYENV_ROOT="$HOME/.pyenv"
PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

case "${TOXENV}" in
        py27)
                curl -O https://bootstrap.pypa.io/get-pip.py
                python get-pip.py --user
                ;;
        py33)
                pyenv install 3.3.6
                pyenv global 3.3.6
                ;;
        py34)
                pyenv install 3.4.6
                pyenv global 3.4.6
                ;;
        py35)
                pyenv install 3.5.3
                pyenv global 3.5.3
                ;;
        py36)
                pyenv install 3.6.1
                pyenv global 3.6.1
                ;;
        py37)
                pyenv install 3.7.2
                pyenv global 3.7.2
                ;;
        py38)
                pyenv install 3.8.0
                pyenv global 3.8.0
                ;;
		py39)
				pyenv install 3.9.0
				pyenv global 3.9.0
				;;
        pypy*)
                pyenv install "$PYPY_VERSION"
                pyenv global "$PYPY_VERSION"
                ;;
        pypy3)
                pyenv install pypy3-2.4.0
                pyenv global pypy3-2.4.0
                ;;
        docs)
                brew install enchant
                curl -O https://bootstrap.pypa.io/get-pip.py
                python get-pip.py --user
                ;;
esac
pyenv rehash
python -m pip install --user virtualenv
python -m virtualenv ~/.venv
source ~/.venv/bin/activate
# This coverage pin must be kept in sync with tox.ini
pip install --upgrade pip
pip install --upgrade wheel
pip install tox
pip install delocate
