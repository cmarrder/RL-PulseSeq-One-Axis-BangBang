#!usr/bin/bash

# ABSOLUTE path to location where directory storing the data for all the runs will be created
pathname="$HOME/Documents/ml/CollectiveAction"
# Name of the new directory which will store the data for all the runs
dirname='data'
# Name of the file to be used as the template for all parameter text files.
# Must be located within the directory specified by $pathname.
paramname='param.txt'
echo $pathname/$dirname
mkdir -p $pathname/$dirname


########## Make array of values using Tmin and Tmax
#readarray -t values < $pathname/'tvals.txt'

# Initialize built-in SECONDS variable so we can time the duration of this program.
SECONDS=0

# Set number of independent learning runs to do.
nRun=2

# Iterate over indices
#for k in ${!values[@]}; do
for k in $(seq 0 $((nRun-1))); do
    (
        echo "starting task $k"

	# Create a name for the output directory which will store this run's data.
	# The name for the 99th run will look like test_00099
	printf -v name "$pathname/$dirname/job_%05d" $k
	mkdir -p $name

        # Replace value for output directory name in param.txt with name and temperature in param.txt with the kth element of values array.
	# Using symbol : as the regex delimiter for clarity. This is because we will be manipulating strings containing symbol / ,
	# which is what we would normally use as the delimeter.
	sed -i "s:noiseParam2 .*:noiseParam2 ${values[$k]}:g" $pathname/$paramname
	sed -i "s:oDir .*:oDir $name:g" $pathname/$paramname
	
        # Run learning code using the param file copy we just modified.
	$pathname/build/learn $name/$paramname
    )

done

echo "All agents done."

# Calculate duration of program
duration=$SECONDS
echo "$(($duration / 3600)) hours, $((($duration / 60) % 60)) minutes and $(($duration % 60)) seconds elapsed."

#echo "Making plots."
#mkdir -p temp_scan_plots
#python testing/plot_tools.py
