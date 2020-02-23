#!/bin/bash


python -c "import ignite.contrib.experimental"
res=$?

if [ "$res" -eq "1" ]; then
    echo "Install experimental ignite"
    pip install --upgrade git+https://github.com/vfdev-5/ignite.git@experimental
fi
