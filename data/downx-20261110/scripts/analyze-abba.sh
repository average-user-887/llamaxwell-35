#!/bin/bash
# usage: analyze-abba.sh raw/modeX/abba.txt ; mean/median per arm, per-cycle delta (A=dx0 mean of 2, B=dx1 mean of 2)
sort_med(){ sort -n | awk '{a[NR]=$1}END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
f=$1
for d in 0 1; do v=$(grep "dx=$d " $f | sed 's/.*tg128=//'); echo "dx=$d n=$(echo "$v"|wc -l) mean=$(echo "$v"|awk '{s+=$1}END{printf "%.4f",s/NR}') median=$(echo "$v"|sort_med) min=$(echo "$v"|sort -n|head -1) max=$(echo "$v"|sort -n|tail -1)"; done
for c in 1 2 3 4 5; do awk -F'[= ]' -v c=$c '$2==c&&$4==0{a+=$NF;na++}$2==c&&$4==1{b+=$NF;nb++}END{printf "cycle %d: dx0=%.3f dx1=%.3f delta=%+.3f (%+.2f%%)\n",c,a/na,b/nb,b/nb-a/na,(b/nb/(a/na)-1)*100}' $f; done
