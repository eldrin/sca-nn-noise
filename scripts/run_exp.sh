model="$1"
dataroot="$2"
nfolds=10

for noise in 0. 0.25 0.5 0.75 1
do
    for size in full 16384 8192 4096 2048 1024
    do
        for fold in $(seq 1 $nfolds)
        do
            echo "[model/dataset: $model] [noise level: $noise] [n_traces: $size]"
            python scripts/train.py \
                "$dataroot/datasets/" \
                "$dataroot/results/" \
                $model \
                $size \
                $noise
        done
    done
done
