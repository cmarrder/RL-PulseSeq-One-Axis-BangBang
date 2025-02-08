#!usr/bin/bash

# This code creates the directory trees needed for a parameter search of the eta parameter in our action functions.
# Given a path, this code will create many jobs, each of which is associated with a single eta value.
# Because of the stochastic behavior of the learning code, we expect some variance in the performance of the learning
# for a given eta.
#
# We repeat the learning code for a given eta value multiple times so we can get the average performance later.
# Each job directory has many subdirectories in it, known as runs, each of which correspond to an identical run
# of the learning code. 


# Initialize built-in SECONDS variable so we can time the duration of this program.
SECONDS=0


# ABSOLUTE path to location where directory storing the data for all the runs will be created
pathname="$HOME/Documents/ml/CollectiveAction"
# Name of the new directory which will store the data for all the runs
dirname='data'
# Name of the file to be used as the template for all parameter text files.
# Must be located within the directory specified by $pathname.
paramname='param.txt'
echo $pathname/$dirname
mkdir -p $pathname/$dirname


readarray -t values < $pathname/'etavals.txt'


# Iterate over indices for each value in values.
for j in ${!values[@]}; do
    (
        echo "starting task $j"

	# Create a name for the output directory which will store this run's data.
	# The name for the 99th job will look like job_00098
	printf -v jobname "$pathname/$dirname/job_%05d" $j
	mkdir -p $jobname

        # Replace value for etaN in param.txt with the jth element of values array.
	# Using symbol : as the regex delimiter for clarity. This is because we will be manipulating strings containing symbol / ,
	# which is what we would normally use as the delimeter.
	sed -i "s:etaN .*:etaN ${values[$j]}:g" $pathname/$paramname
	
        # Make runs.
        for ((k = 0 ; k < 10 ; k++ )); do
	        # Create a name for the output directory which will store this run's data.
	        # The name for the 99th job will look like job_00098
	        printf -v runname "$jobname/run_%05d" $k
	        mkdir -p $runname

                # Replace value for output directory name in param.txt with jobname
	        # Using symbol : as the regex delimiter for clarity. This is because we will be manipulating strings containing symbol / ,
	        # which is what we would normally use as the delimeter.
	        sed -i "s:oDir .*:oDir $runname:g" $pathname/$paramname

                # Run learning code
	        $pathname/build/learn
	done

    )

done

echo "All agents done."

# Calculate duration of program
duration=$SECONDS
echo "$(($duration / 3600)) hours, $((($duration / 60) % 60)) minutes and $(($duration % 60)) seconds elapsed."
