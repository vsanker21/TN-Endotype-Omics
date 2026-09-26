#!/usr/bin/env bash
# First read header of each GSE186505 R1 FASTQ (instrument:run:flowcell:lane) streamed from ENA without full download.
set -u
BASE=~/tn_gse186505
OUT="$BASE/fastq_headers.tsv"
printf "run_accession\tsample_alias\theaders_first4\n" > "$OUT"
tail -n +2 "$BASE/ena.tsv" | while IFS=$'\t' read -r run gsm ftp md5; do
  r1="${ftp%%;*}"
  h=$(curl -s --max-time 60 -r 0-262143 "https://$r1" | gzip -dc 2>/dev/null | awk 'NR%4==1' | head -200 | cut -d' ' -f2 | cut -d: -f1-4 | sort | uniq -c | sort -rn | head -4 | awk '{printf "%s(%s) ", $2, $1}')
  printf "%s\t%s\t%s\n" "$run" "$gsm" "$h" >> "$OUT"
done
cat "$OUT"
