#!/bin/bash

pip3 install pdoc3
pdoc --force --template-dir templates -o docs --html mimir
mv ./docs/mimir/* ./docs/
rm -r ./docs/mimir