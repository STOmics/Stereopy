#!/bin/bash

conda activate stereopy-doc

rm -rf ./build/*

make html
