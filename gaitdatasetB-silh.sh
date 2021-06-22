#!/bin/sh

for i in `seq 124`
do
  if [ $i -lt 10 ]; then
    tar -zxvf ./GaitDatasetB-silh/00$i.tar.gz -C CASIA-B
  elif [ $i -lt 100 ]; then
    tar -zxvf ./GaitDatasetB-silh/0$i.tar.gz -C CASIA-B
  else
    tar -zxvf ./GaitDatasetB-silh/$i.tar.gz -C CASIA-B
  fi
done
