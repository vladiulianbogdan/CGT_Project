trap "exit" INT TERM ERR
trap "kill 0" EXIT

GEN=100000

for lossRich in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1.0
    do
      for inRound in 1 2 3 4
        do
          for lossPoor in 0.5 1
              do
                  python simulation.py $GEN 4 2 100 $inRound $lossPoor $lossRich 1000 1 4 3 1 10 figure3_ &
              done
        done
    done

wait

kill 0
