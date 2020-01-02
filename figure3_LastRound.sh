trap "exit" INT TERM ERR
trap "kill 0" EXIT

GEN=100000

for lossRich in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1.0
    do
      for lossPoor in 0.5 1
          do
              python simulation.py $GEN 4 2 100 3 $lossPoor $lossRich 1000 1 4 3 1 10 figure3_ &
          done
      wait
    done

wait

kill 0
