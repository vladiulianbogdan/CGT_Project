trap "exit" INT TERM ERR
trap "kill 0" EXIT

GEN=100000

for alphaConfiguration in "1 1" "1 0.8" "0.5 0.8" "0.5 0.5"
    do
        set -- $alphaConfiguration

        for riskInRound in 1 2 3 4
        do
            python3 simulation.py $GEN 4 2 100 $riskInRound $1 $2 1000 1 4 3 1 10 figure4_ &
        done
        wait
    done

wait

kill 0