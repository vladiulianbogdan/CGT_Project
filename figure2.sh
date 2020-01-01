trap "exit" INT TERM ERR
trap "kill 0" EXIT

GEN=10000

for riskFunctionParameters in "1 1" "2 10" "1 10" "3 7"
    do
        set -- $riskFunctionParameters

        for rounds in 1 2 4
        do
            python simulation.py $GEN $rounds 2 100 1 1 1 1000 1 1 $1 0 $2 figure2_ &
        done
        wait
    done

wait

kill 0
