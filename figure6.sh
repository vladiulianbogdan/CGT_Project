trap "exit" INT TERM ERR
trap "kill 0" EXIT

GEN=100000

for riskFunctionParameters in "1 1" "2 10" "1 10" "3 7"
    do
        set -- $riskFunctionParameters

        python simulation.py $GEN 4 2 100 1 1 1 1000 1 1 $1 0 $2 figure2_inv2_ &
    done

wait

kill 0
