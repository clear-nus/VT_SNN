#!/usr/bin/env bash
set -euo pipefail


if [ $# -eq 0 ]; then
    echo "Slip lite is being downloaded ..."
    ./gdrivedl.py https://drive.google.com/file/d/1VBCwDNwjRqRMQ4iPHo8WRh9n5g92nzt6/view?usp=sharing
fi


if [ "$1" == "preprocessed" ]; then
    echo "Slip preprocessed data is being downloaded ..."
    ./gdrivedl.py https://drive.google.com/file/d/1ODFWBdCMq6NT1O4jDI1zxBc-3flF8U7h/view?usp=sharing
else
    echo "Slip lite is being downloaded ..."
    ./gdrivedl.py https://drive.google.com/file/d/1VBCwDNwjRqRMQ4iPHo8WRh9n5g92nzt6/view?usp=sharing
fi