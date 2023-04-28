#!/bin/bash

if [[ $# -ne 1 ]]
then
  echo "The fitzpatrick17k csv must be the first and only argument"
  exit 1
fi

nlines=$(( $(wc -l $1 | cut -d ' ' -f 1) - 1 ))
i=0

mkdir -p images
mkdir -p failed_images

tail -n +2 $1 | while read -r line
do
  md5hash=$(cut -d ',' -f 2 <<< $line)
  url=$(cut -d ',' -f 8 <<< $line)
  filename="images/${md5hash}.jpg"

  if ! [[ -f $filename ]]
  then
    wget --quiet -O $filename $url && echo $md5hash $filename >> md5sums.txt
  fi

  (( i += 1 ))
  echo -ne $i / $nlines '\r'
done

echo -ne '\n'

for bad_file in $(md5sum -c md5sums.txt 2> /dev/null | grep FAILED | cut -d ':' -f 1)
do
  mv $bad_file failed_images
done
