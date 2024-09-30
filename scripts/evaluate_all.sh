shopt -s nullglob

regex="${1:-.*}"

for b in data/*; do
    if [[ ! -d $b/predictions ]]; then
        continue
    fi
    echo `basename $b`
    for f in $b/predictions/*.txt; do
        if [[ ! $f =~ $regex ]]; then
            continue
        fi
        echo `basename $f`
        python scripts/evaluate.py \
            --input $b/test_input.jsonl \
            --target $b/test_target.jsonl \
            --prediction $f \
            --kg wikidata \
            --allow-subset \
            --prediction-format jsonl \
            ${@:2}
        echo ""
    done
    echo "---------"
done

shopt -u nullglob
