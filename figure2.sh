trap "exit" INT TERM ERR
trap "kill 0" EXIT

GEN=100000

for riskFunctionParameters in "1 1" "2 10" "1 10" "3 7"
    do
        set -- $riskFunctionParameters

        for riskInRound in 1 2 3 4
        do
            for rounds in 1 2 4
            do
                python simulation.py $GEN $rounds 2 100 $riskInRound 1 1 1000 1 1 $1 0 $2 figure2 &
            done
        done

        wait
    done

wait

kill 0
