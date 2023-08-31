#!/bin/zsh

OS_NAME="macos"
COMPILER="clang"
CONDA_PREFIX=$(conda info --base)

if [[ "$OS_NAME" == "macos" && "$COMPILER" == "clang" ]]; then
    for LIBOMP_ALIAS in libgomp.dylib libiomp5.dylib libomp.dylib
    do
        sudo ln -sf "$(brew --cellar libomp)"/*/lib/libomp.dylib $CONDA_PREFIX/lib/$LIBOMP_ALIAS
    done
fi

