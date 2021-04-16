#!/bin/bash
set -e -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

# Install a system package required by our library
yum install -y freetype-devel
yum install -y libpng-devel
if [ `getconf LONG_BIT` = "32" ]
then
    yum install -y pkg-config
fi



# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install -r /io/dev-requirements.txt
    "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install pyedflib --no-index -f /io/wheelhouse
    (cd "$HOME"; "${PYBIN}/pytest" --pyargs pyedflib)
done
