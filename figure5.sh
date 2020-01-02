trap "exit" INT TERM ERR
trap "kill 0" EXIT

GEN=5

for riskcurve in 1 2 3
	do	
	for rounds in 1 4
		do	
		for lambdaConfiguration in 0.01 0.1 1.0 10.0 100.0 
			do
			for alphaConfiguration in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
				do
					python3 simulation.py $GEN $rounds 2 100 1 $alphaConfiguration $alphaConfiguration 1000 2 2 $riskcurve 1 $lambdaConfiguration figure5_ &
				done
			wait
			done
		wait
		done
	wait
	done
	
kill 0