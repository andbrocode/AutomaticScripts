#!/bin/bash

year=$(date -d '-1 day' +%Y)

doy=$(date -d '-1 day' +%j)

doy_old=$(date -d '-3 day' +%j)

datetime=$(date -d "-1days" +%Y-%m-%d)

ring="Z"
ring=$1

path_remote="/import/freenas-ffb-01-data/romy_archive/${year}/BW/DROMY/FJ${ring}.D/"
path_local="/home/brotzer/archive/${year}/BW/DROMY/FJ${ring}.D/"

file="BW.DROMY..FJ${ring}.D.${year}.${doy}"


if [ ! -f ${path_local}${file} ]; then
    echo -e "copying data ...\n"
    scp brotzer@kilauea:${path_remote}${file} ${path_local}
fi

echo -e "running sagnac processing...\n"
python3 ~/scripts/autodata_SagnacFrequency_MPrandom_daily_mod.py ${ring} ${datetime}

## remove old data
if [ -f ${path_local}BW.DROMY..FJ${ring}.D.${year}.${doy_old} ]; then
    rm ${path_local}BW.DROMY..FJ${ring}.D.${year}.${doy_old}
fi


## End of File
