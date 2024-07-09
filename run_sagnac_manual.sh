#!/bin/bash

#year=$(date -d '-1 day' +%Y)

#doy=$(date -d '-1 day' +%j)

#doy_old=$(date -d '-3 day' +%j)

#datetime=$(date -d "-1days" +%Y-%m-%d)


ring=$1

dat=$2

year=$(date -d "${dat}" +%Y)

doy1=$(date -d "${dat}" +%j)
doy2=$(date -d "${dat} +1days" +%j)
doy3=$(date -d "${dat} -1days" +%j)

path_remote="/import/freenas-ffb-01-data/romy_archive/${year}/BW/DROMY/"
path_local="/home/brotzer/archive/${year}/BW/DROMY/"

file1="FJ${ring}.D/BW.DROMY..FJ${ring}.D.${year}.${doy1}"
file2="FJ${ring}.D/BW.DROMY..FJ${ring}.D.${year}.${doy2}"
file3="FJ${ring}.D/BW.DROMY..FJ${ring}.D.${year}.${doy3}"
file4="F1V.D/BW.DROMY..F1V.D.${year}.${doy3}"
file5="F2V.D/BW.DROMY..F2V.D.${year}.${doy3}"

for file in $file1 $file2 $file3 $file4 $file5; do
        #echo $file
	if [ ! -f ${path_local}${file} ]; then
    		echo -e "copying data ...\n"
    		scp -i /home/brotzer/.ssh/id_ed25519_kilauea brotzer@kilauea:${path_remote}${file} ${path_local}${file}
	fi
done

echo -e "running sagnac processing...\n"
python3 ~/scripts/autodata_SagnacFrequency_MPrandom_daily_mod.py ${ring} ${dat}

## remove old data
for file in $file1 $file2 $file3 $file4 $file5; do

	if [ -f ${path_local}${file} ]; then
    		rm ${path_local}${file}
	fi
done

# replace - in date string
dd=$(sed 's/-//g' <<< ${dat})

# copy outpuf file back to kilauea
#scp  -i /home/brotzer/.ssh/id_ed25519_kilauea /home/brotzer/archive/FJ${ring}*${dd}*.pkl brotzer@kilauea:/import/freenas-ffb-01-data/romy_autodata/${year}/R${ring}/

## End of File
