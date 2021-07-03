#!/bin/sh

for i in `seq 124`
do
  if [ $i -lt 10 ]; then
    mv ../CASIA-B/00$i ../CASIA-B_train
  elif [ $i -lt 75 ]; then
    mv ../CASIA-B/0$i ../CASIA-B_train
  elif [ $i -lt 100 ]; then
    mv ../CASIA-B/0$i ../CASIA-B_test
  elif [ $i -lt 125 ]; then
    mv ../CASIA-B/$i ../CASIA-B_test
  fi
done
