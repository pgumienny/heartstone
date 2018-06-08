#!/bin/bash

if [[ "$#" -ne 2 ]]; then
    echo "Usage: $0 in_file out_file"
fi

in="$1"
out="$2"

sort "$in" | uniq -c | sort -nr > "$out"

exit 0
