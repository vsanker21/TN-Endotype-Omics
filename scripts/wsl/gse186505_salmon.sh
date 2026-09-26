#!/usr/bin/env bash
# Re-quantify GSE186505 (PRJNA774204) whole-blood RNA-seq from raw reads with salmon.
# Downloads run in parallel; each sample is quantified once both mates are downloaded and md5-verified,
# after which the FASTQ files are deleted.
set -uo pipefail
BASE=~/tn_gse186505
IDX=~/meningioma_gtex_v8_requant/ref/salmon_index   # GENCODE v26 transcriptome, salmon 1.10.2, k=31
mkdir -p "$BASE"/{fastq,quant,logs}
source ~/miniforge3/etc/profile.d/conda.sh
if ! conda env list | grep -q '^salmon110 '; then
  mamba create -y -n salmon110 -c conda-forge -c bioconda salmon=1.10.3 > "$BASE/logs/env.log" 2>&1
fi
conda activate salmon110
salmon --version

curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA774204&result=read_run&fields=run_accession,sample_alias,fastq_ftp,fastq_md5&format=tsv" > "$BASE/ena.tsv"
cat "$BASE/ena.tsv"

download_run() {
  local run=$1 ftp=$2 md5=$3 base=$4
  local u1=${ftp%%;*} u2=${ftp##*;} m1=${md5%%;*} m2=${md5##*;}
  for pair in "1 $u1 $m1" "2 $u2 $m2"; do
    set -- $pair
    local out="$base/fastq/${run}_$1.fastq.gz"
    for attempt in 1 2 3; do
      wget -q -c -O "$out" "https://$2" && [ "$(md5sum "$out" | cut -d' ' -f1)" = "$3" ] && break
      rm -f "$out"
    done
  done
  [ -f "$base/fastq/${run}_1.fastq.gz" ] && [ -f "$base/fastq/${run}_2.fastq.gz" ] && touch "$base/fastq/${run}.ready"
}
export -f download_run

tail -n +2 "$BASE/ena.tsv" | while IFS=$'\t' read -r run alias ftp md5; do
  [ -f "$BASE/quant/$run/quant.sf" ] || echo "$run $ftp $md5"
done | xargs -P 10 -L 1 bash -c 'download_run "$0" "$1" "$2" '"$BASE" > "$BASE/logs/download.log" 2>&1 &

N=$(tail -n +2 "$BASE/ena.tsv" | wc -l)
while :; do
  done_n=$(ls "$BASE"/quant/*/quant.sf 2>/dev/null | wc -l)
  [ "$done_n" -ge "$N" ] && break
  for r in "$BASE"/fastq/*.ready; do
    [ -e "$r" ] || continue
    run=$(basename "$r" .ready)
    salmon quant -i "$IDX" -l A -1 "$BASE/fastq/${run}_1.fastq.gz" -2 "$BASE/fastq/${run}_2.fastq.gz" \
      -p 16 --validateMappings --gcBias --seqBias -o "$BASE/quant/$run" > "$BASE/logs/$run.salmon.log" 2>&1 \
      && rm -f "$BASE/fastq/${run}"_*.fastq.gz "$r"
    echo "$(date +%T) quantified $run ($(ls "$BASE"/quant/*/quant.sf | wc -l)/$N)"
  done
  if ! pgrep -f download_run > /dev/null && ! ls "$BASE"/fastq/*.ready > /dev/null 2>&1; then
    echo "downloads finished; quantified $(ls "$BASE"/quant/*/quant.sf 2>/dev/null | wc -l)/$N"; break
  fi
  sleep 30
done
echo ALL_DONE
