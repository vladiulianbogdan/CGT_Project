trap "exit" INT TERM ERR
trap "kill 0" EXIT

GEN=100000

for riskFunctionParameters in "1 1" "2 10" "3 7" "1 10" 
    do  
    	for rounds in 1 2 4
    		do 
                set -- $riskFunctionParameters
        		for alpha in 0.0 0.2 0.4 0.6 0.8 1.0
        			do
            			python3 simulation.py $GEN $rounds 2 100 1 $alpha $alpha 1000 1 1 $1 0 $2 figure1_V1_ &
        			done
        			wait
        	done 
    done

wait

kill 0