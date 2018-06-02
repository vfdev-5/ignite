#!/usr/bin/env bash

# . /remote/anaconda_token || true

set -e

if [ -z "$ANACONDA_TOKEN" ]; then
    echo "ANACONDA_TOKEN is unset. Please set it in your environment before running this script";
    exit 1
fi

ANACONDA_USER=vfdev-5
conda config --set anaconda_upload no

set -e
IGNITE_BUILD_VERSION="0.1.0"
IGNITE_BUILD_NUMBER=1

rm -rf ignite-src
git clone https://github.com/vfdev-5/ignite ignite-src
pushd ignite-src
git checkout v$IGNITE_BUILD_VERSION
popd

export PYTORCH_IGNITE_BUILD_VERSION=$VISION_BUILD_VERSION
export PYTORCH_IGNITE_BUILD_NUMBER=$VISION_BUILD_NUMBER

time conda build -c $ANACONDA_USER --no-anaconda-upload --python 2.7 ignite-$IGNITE_BUILD_VERSION
time conda build -c $ANACONDA_USER --no-anaconda-upload --python 3.5 ignite-$IGNITE_BUILD_VERSION
time conda build -c $ANACONDA_USER --no-anaconda-upload --python 3.6 ignite-$IGNITE_BUILD_VERSION

set +e

unset PYTORCH_BUILD_VERSION
unset PYTORCH_BUILD_NUMBER
unset PYTORCH_IGNITE_BUILD_VERSION
unset PYTORCH_IGNITE_BUILD_NUMBER
