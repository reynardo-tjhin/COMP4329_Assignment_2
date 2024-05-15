#!/bin/bash

# remove any unecessary temporary files
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$|instance)" | xargs rm -rf

# for (( i=0; i<10; i++ ))
# do
#     TEST=$((39990+$i))
#     rm "$TEST".test
# done

# for (( i=0; i<10; i++ ))
# do
#     TEST=$((40000+$i))
#     touch "$TEST".test
# done

# for (( i=0; i<10; i++ ))
# do
#     TEST=$((40000+$i))
#     rm "$TEST".test
# done
