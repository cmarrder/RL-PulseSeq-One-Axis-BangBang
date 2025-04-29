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
data_dir_name="data"
# Name of the build directory
build_dir="$pathname/build"
# Name of the file to be used as the template for all parameter text files.
# Must be located within the directory specified by $pathname.
paramname='param.txt'

echo -e "\nWill be writing data to the path:"
echo $pathname/$data_dir_name
mkdir -p $pathname/$data_dir_name

# Read values of pulse numbers into array called values.
readarray -t values < $pathname/'pulse_numbers.txt'

cd $build_dir

#element_count=${#values[@]}
#total_runs=$((element_count * element_count))

# Iterate over indices for each value in values.
for N in ${values[@]}; do
    (
	# Create a name for the output directory which will store this run's data.
	# The name for N = 98  will look like N_098
	printf -v jobname "$pathname/$data_dir_name/N_%03d" $N
	mkdir -p $jobname

        # Replace value for eta1 in param.txt with the jth element of values array.
	# Using symbol : as the regex delimiter for clarity. This is because we will be manipulating strings containing symbol / ,
	# which is what we would normally use as the delimeter.
	sed -i "s:const int nPulse = .*:const int nPulse = $N;:" $pathname/include/Sequence.hpp

	# Compile code
        echo -e "\nStarting to compile C++ code."
        make
        echo "Finished compiling C++ code."
	
        # Make runs.
        for J in ${values[@]}; do
                
	        # Create a name for the output directory which will store this run's data.
	        # The name for J = 98 will look like J_098
	        printf -v runname "$jobname/J_%03d" $J
	        mkdir -p $runname

                # Replace value for output directory name in param.txt with jobname
	        # Using symbol : as the regex delimiter for clarity. This is because we will be manipulating strings containing symbol / ,
	        # which is what we would normally use as the delimeter.
	        sed -i "s:oDir .*:oDir $runname:" $pathname/$paramname

		# Replace value of maxJ
	        sed -i "s:maxJ .*:maxJ $J:" $pathname/$paramname

		echo -e "\nStarting N = $N, J = $J\n"

                # Run learning code
	        $pathname/build/learn

		# Print how much time has elapsed so far
                duration=$SECONDS
                echo "$(($duration / 3600)) hours, $((($duration / 60) % 60)) minutes and $(($duration % 60)) seconds elapsed."
	done

    )

done

cd $pathname

echo "All agents done."

# Calculate duration of program
#duration=$SECONDS
#echo "$(($duration / 3600)) hours, $((($duration / 60) % 60)) minutes and $(($duration % 60)) seconds elapsed."


# Next steps:
# Change Param.hpp to add maxJ as a variable.
# Load maxJ in Crystal.hpp
