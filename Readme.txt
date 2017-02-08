Runs with Apache Spark version 1.3.0 and above, as well as OpenCV 2.4.9 and above.

To run tests:
1. Copy the "data" folder to your working directory. It contains example data and metadata
2. Compile the source files using maven "mvn package", and copy the generated jar file (in ./target) to your working directory
3. Specify correct working directory in the script "script4mn_job", as well as your parameters of choice.
4. Login to BSC, load the spark module and run the test!
	module load spark4mn
	spark4mn script4mn

