package sparkBoVW;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FileUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.SparkConf;

import org.opencv.core.Core;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector; 

import boWHistograms.BoWHistograms;
import boWIDE.BoWIDE;
import boWModel.BoWModel;
import sVM.SVMForBoW;
import scala.Tuple2;
import utils.FilingUtils;

/**
 * 
 * @author Sana Imtiaz
 * <p><p>
 * Main Class for performing image recognition using Bag of Words in Spark. Uses OpenCV library on backend for some image processing functions 
 *
 */
public class Main {

	
	public static void main(String[] args) throws Exception{       
		
		
		long startTime=0;
		
		// Load the OpenCV runtime library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		// Set configuration parameters for Apache Spark on driver node and create Spark Context in Java
		SparkConf conf = new SparkConf().setAppName("BoW Test");
		//conf.set("spark.driver.memory","2g").set("spark.broadcast.compress","true");
		//conf.set("spark.task.maxFailures","4");
		//conf.set("spark.io.compression.codec","snappy").set("spark.serializer","org.apache.spark.serializer.KryoSerializer");        
	    JavaSparkContext sc = new JavaSparkContext(conf); 
		
	
		//------------------ Initializations --------------------
		//String workingDir = "/Users/mouna/these/bow-master/SparkVersion/data";
	    
	    //String workingDir = "/gpfs/projects/bsc31/bsc31914/tests/data";//args[0];
		String workingDir = "/Users/mouna/these/bow-master/SparkVersion/spam";
	    
	    //String pathToImageDir = "/gpfs/projects/bsc31/bsc31914/tests/data/images";//workingDir+"/images";
	    String pathToImageDir = workingDir+"/images";
		
	    String pathsToTrainingImages = workingDir+"/trainingImagePaths.txt";
		//String pathsToTrainingImages = "/gpfs/projects/bsc31/bsc31914/tests/data/trainingImagePaths.txt";//workingDir+"/trainingImagePaths.txt";
		
	    //String pathsToTestingImages = "/gpfs/projects/bsc31/bsc31914/tests/data/testingImagePaths.txt";//workingDir+"/testingImagePaths.txt";
	    String pathsToTestingImages = workingDir+"/testingImagePaths.txt";
		
	    String pathToMetadataDir = workingDir+"/metadata";
	    //String pathToMetadataDir = "/gpfs/projects/bsc31/bsc31914/tests/data/metadata";//workingDir+"/metadata";
		
	    String pathsToTrainingMetadata = workingDir+"/trainingMetadataPaths.txt";
		//String pathsToTrainingMetadata = "/gpfs/projects/bsc31/bsc31914/tests/data/trainingMetadataPaths.txt";//workingDir+"/trainingMetadataPaths.txt";
		
	    String pathsToTestingMetadata = workingDir+"/testingMetadataPaths.txt";
	    //String pathsToTestingMetadata = "/gpfs/projects/bsc31/bsc31914/tests/data/testingMetadataPaths.txt";//workingDir+"/testingMetadataPaths.txt";
	
	    //	String pathToImageDescriptors = workingDir+"/imageDescriptors";
		//String pathToKmeansModel = workingDir+"/pathToKmeanModel.txt";
	//	String pathToKmeansModel = workingDir+"/BoVWDictionary/";
	//	String pathToSVMTrainingData = workingDir+"/SVMData/svmTrainingData.txt";
	///	String pathToSVMTestingData = workingDir+"/SVMData/svmTestingData.txt";		
	//	String pathToSVMModel = workingDir+"/SVMModel/";
	//	String pathToLabelsMap = workingDir+"/SVMData/labelsMap.txt";
		
		//String pathToLog = "/gpfs/projects/bsc31/bsc31914/tests/data/log.txt";
		String pathToLog = workingDir+"/log.txt";
		int minPartitions = 6;
		
		//---------- Configuration parameters for algorithm ---------------------
		double trainingFraction = (double)0.8;//Double.parseDouble(args[1]); 	
		int numClusters = 150;//Integer.parseInt(args[2]);	// k-means clusters or dictionary words
		int numIterations = 15;//Integer.parseInt(args[3]);	// iterations for k-means	
		int numParRuns = 1;//Integer.parseInt(args[4]);		// number of parallel runs for k-means
		String initMode = "random";//args[5];						// initialization mode for centers
		int numIters = 200;//Integer.parseInt(args[6]);		// iterations for creating SVM Model
		double svmThreshold = -0.4013;//Double.parseDouble(args[7]);		// Threshold for SVM for distinguishing positive samples from the corpus
		int resizeImageToDimension =  256;//Integer.parseInt(args[8]); // Resize all images to this dimension (in pixels)
		boolean gaussianBlur =  false;//Boolean.parseBoolean(args[9]);	// Flag for applying Gaussian blur
		boolean colour = true;//Boolean.parseBoolean(args[10]);		// Flag for using colour information in images
		
	    int resizeImageToRows = resizeImageToDimension;//256
		int resizeImageToCols = resizeImageToDimension;//256
		int detectorType = FeatureDetector.SIFT;
		int extractorType;
		if (colour)
			extractorType = DescriptorExtractor.OPPONENT_SIFT;
		else 
			extractorType = DescriptorExtractor.SIFT;
		
		BufferedWriter logwriter = new BufferedWriter(new FileWriter(pathToLog, true));//append
		
		//------------------ Write input paths to text files -------------
		
		FilingUtils.writePathsToFiles (pathToImageDir, pathsToTrainingImages, pathsToTestingImages, pathToMetadataDir, 
				pathsToTrainingMetadata, pathsToTestingMetadata, trainingFraction);
		
		/**************************** Main Program *************************/
		System.out.print("\nParameters: "+trainingFraction+"/"+(1-trainingFraction)+" train/test ratio, "+numClusters+" words in "
				+numIterations+" iterations, "+numIters+" iterations for creating SVM Model, "+svmThreshold+" as SVM threshold");
		/*if (gaussianBlur)
			System.out.print(" with blur");
		if (colour)
			System.out.print(" and colour");
		System.out.println();
		*/
		//---------- Create training, testing and metadata RDDs for images------
		
		startTime = System.currentTimeMillis();
		JavaRDD<String> trainingImagesList = sc.textFile(pathsToTrainingImages,6);// minPartitions);
		logwriter.write("Creating training RDDs in "+ ((double)(System.currentTimeMillis() - startTime))/1000.0 + " seconds");
		logwriter.newLine();
		logwriter.write("-- Training Images: "+trainingImagesList.count());
		logwriter.newLine();
		
		
		
		
		JavaRDD<String> testingImagesList = sc.textFile(pathsToTestingImages, 6);// minPartitions);
		logwriter.write("-- Testing Images: "+testingImagesList.count());
		logwriter.newLine();
		
		JavaRDD<String> testingMetadataPaths =  sc.textFile(pathsToTestingMetadata, 6);// minPartitions);
		
		
		//----------------Train the BoW Model and optionally save to Disk ------------------
		
		
		
		// Get SIFT descriptors for all images and write them to disk
		startTime = System.currentTimeMillis();
		JavaRDD<Vector> imageDescriptors = trainingImagesList.flatMap(new
				boWModel.ExtractDescriptors(resizeImageToRows,
				resizeImageToCols, detectorType, extractorType, gaussianBlur, colour)).cache();
		
		logwriter.write("Extract descriptors for training images in "+ ((double)(System.currentTimeMillis() - startTime))/1000.0 + " seconds");
		logwriter.newLine();
		logwriter.write("-- Total descriptors: "+imageDescriptors.count());
		logwriter.newLine();
		
		//imageDescriptors.saveAsTextFile(workingDir+"/ResImageDes.txt");
		imageDescriptors.repartition(minPartitions).cache();
		
		//File descriptorsDir = new File(pathToImageDescriptors);
		//if (descriptorsDir.exists()){
			//FileUtils.cleanDirectory(descriptorsDir);
			//FileUtils.deleteDirectory(descriptorsDir);
		//}
		
		
		// ---------------- Create the BoW dictionary (KMeansModel): --------------------
		//cluster the descriptors into k common descriptors (words) using KMeans  
		
		BoWModel aBoWModel = BoWModel.getSingleton();
		
		
		
		numClusters=100; //dictionary words
		numIterations=15; //iterations for kmeans
		numParRuns=1;
		//initMode=
		logwriter.write("number iterations kmeans " + numIterations);
		logwriter.newLine();
		logwriter.write("number runs " + numParRuns);
		logwriter.newLine();
		KMeansModel dictionary = KMeans.train(imageDescriptors.rdd(), numClusters, numIterations,numParRuns,KMeans.K_MEANS_PARALLEL());
		logwriter.write("BoW dictionary with "+dictionary.k()+" words created in "+ ((double)(System.currentTimeMillis() - startTime))/1000.0 + " seconds");
		logwriter.newLine();
		//KMeansModel dictionary = aBoWModel.createModel(imageDescriptorsFromDisk, numClusters, numIterations, numParRuns, initMode);
		//KMeansModel dictionary = aBoWModel.createModel(imageDescriptors, numClusters, numIterations);
	
		//logwriter.write("Within Set Sum of Squared Errors = " + dictionary.computeCost(imageDescriptors.rdd()));
		//logwriter.newLine();
		
		
		/*logwriter.write("Centers clusters");
	   
		for (Vector center : dictionary.clusterCenters()) {
	    	logwriter.write(" , " + center.size());
	    	logwriter.newLine();
	    }
	    
	    List<Vector> vectors = imageDescriptors.collect();
	    logwriter.write(vectors.size());
	    for(Vector vector: vectors){
	    	logwriter.write("cluster "+dictionary.predict(vector) +" ");//+vector.toString());
	    	logwriter.newLine();
	    }*/
	      
	   
		// Save BoW dictionary to disk
	 
	    //aBoWModel.saveToDisk(dictionary, sc.sc(), pathToKmeansModel);
	    //System.out.println("BoW Dictionary saved to:"+pathToKmeansModel);
	  	
	    
	    
	    //---------------------- Create BoW training data for SVM -----------------
	   /* BoWIDE aBoWIDE = BoWIDE.getSingleton();
	    
	    // Extract SIFT descriptors for each image separately
	    startTime = System.currentTimeMillis();
		JavaPairRDD<String,Vector> imageDescriptorsWithIDs = trainingImagesList.flatMapToPair(new
				boWIDE.ExtractDescriptorsWithImgNames(resizeImageToRows,
				resizeImageToCols, detectorType, extractorType, gaussianBlur, colour));
	    
		logwriter.write("Extract descriptors for LABELED training images in "+ ((double)(System.currentTimeMillis() - startTime))/1000.0 + " seconds");
		logwriter.newLine();
		logwriter.write("-- Total descriptors: "+imageDescriptorsWithIDs.count());
		logwriter.newLine();
		
		// Create training histograms for each image (BoWBags) by matching descriptors with BoW dictionary
		
		startTime = System.currentTimeMillis();
	    JavaPairRDD<String,Vector> idsWithHistograms = aBoWIDE.getBoWBags(imageDescriptorsWithIDs, dictionary);    		
		idsWithHistograms.collect();  
		logwriter.write("Create training histograms for LABELED training images (BoWBags) in "+ ((double)(System.currentTimeMillis() - startTime))/1000.0 +" seconds");
		logwriter.newLine();
		
		startTime = System.currentTimeMillis();
		// Extract category tags for images from respective metadata files
		JavaRDD<String> trainingMetadataPaths = sc.textFile(pathsToTrainingMetadata, minPartitions);
	    JavaPairRDD<String,String> idsWithTags = BoWHistograms.getTags(trainingMetadataPaths);
		
		// Create category names to numeric labels map
	    Map<String, Double> categoryLabels = BoWHistograms.createLabelsFromTags(idsWithTags);
	   
	    logwriter.write("Create Labels/tags in "+ ((double)(System.currentTimeMillis() - startTime))/1000.0 +" seconds");
		logwriter.newLine();
		logwriter.write("--- Categories number "+categoryLabels.size() + " Content" + categoryLabels.toString());
	    logwriter.newLine();
	    
	    // Create the multiclass BoW training histograms
	    
	    startTime = System.currentTimeMillis();
	    JavaPairRDD<Double, Vector> rawTrainingHistograms = BoWHistograms.getRawHistograms(idsWithHistograms, idsWithTags, categoryLabels);
	    rawTrainingHistograms.cache();
	    
	    
	    logwriter.write("Training histograms in "+ ((double)(System.currentTimeMillis() - startTime))/1000.0 +" seconds");
		logwriter.newLine();
		logwriter.write("--- raw training hist count "+rawTrainingHistograms.count());
	    logwriter.newLine();
	     
	    // Save category names to numeric labels map to disk
	    FilingUtils.saveLabelsToDisk(pathToLabelsMap, categoryLabels);
	    
	   // FileUtils.saveHistogramsToFile(rawTrainingHistograms, pathToSVMTrainingData);
	    //System.out.println(trainHist+" training histograms saved to"+pathToSVMTrainingData);

		
		 //---------------------- Create test BoW vectors and save to Disk ------------
		 BoWIDE anotherBoWIDE = BoWIDE.getSingleton();
	     //KMeansModel boVWDictionary = anotherBoWIDE.loadDictionary(sc.sc(), pathToKmeansModel);
	    
	     KMeansModel boVWDictionary = dictionary;
	     
	     // Extract SIFT descriptors for testing images
	    // 
	    JavaPairRDD<String,Vector> testImageDescriptorsWithIDs = testingImagesList.flatMapToPair(new
					boWIDE.ExtractDescriptorsWithImgNames(resizeImageToRows,
					resizeImageToCols, detectorType, extractorType, gaussianBlur, colour));
	     
	     // Create testing histograms for each image (BoWBags) by matching descriptors with BoW dictionary
		 //
	    //JavaPairRDD<String,Vector> testIdsWithHistograms = aBoWIDE.getBoWBags(testImageDescriptorsWithIDs, boVWDictionary);
		 
		 // Extract category tags for images from respective metadata files
		 //
	    //JavaPairRDD<String,String> testIdsWithTags = BoWHistograms.getTags(testingMetadataPaths);
		 
		 // Load category names to numeric labels map from disk
		 //
	    //Map<String, Double> labels = FilingUtils.loadLabelsFromDisk(pathToLabelsMap);	 
		 
		 // Create the multiclass BoW testing histograms
		 //
	    //JavaPairRDD<Double, Vector> rawTestingHistograms = BoWHistograms.getRawHistograms(testIdsWithHistograms, testIdsWithTags, labels);
		 //rawTestingHistograms.cache();
		 
		 //
	    //System.out.println("Created "+ rawTestingHistograms.count()+" raw testing histograms for the classifier");
		 //FilingUtils.saveHistogramsToFile(rawTestingHistograms, pathToSVMTestingData);
		 //System.out.println(testHist+" testing histograms saved to"+pathToSVMTestingData);
		 
		 
		 /********************************* SVM Part ***********************************/  
		 
	    
		 /* Read Labels from disk, get category keys, Create a binary classifier for each category*/
		 /*
		 for (String tag: labels.keySet()){
		 
			 Double currLabel = labels.get(tag);
			 System.out.println("\nProcessing: "+tag+"...");
			 
			 //------------------------ Train the SVM and possibly save to Disk ---------------------
			 // Create binary histograms for this class
			 JavaPairRDD<Double, Vector> trainingHists =  SVMForBoW.genClassHistograms(sc, rawTrainingHistograms, currLabel);
			
			 // Create SVM training data from these histograms
			 JavaRDD<LabeledPoint> trainingData =  trainingHists.map(t-> new LabeledPoint(t._1.doubleValue(),t._2));
			 			
			 // Create an SVM model and train it!
			 SVMModel model = new SVMModel(null, 0.5); // Create a new SVM Model
			 long time = System.currentTimeMillis();
			 model = SVMWithSGD.train(trainingData.rdd(), numIters);
			 double svmTrainTime = (double) (System.currentTimeMillis()- time);
			 System.out.println("Trained the SVM for "+tag+" in "+numIters+" iterations in "+svmTrainTime/1000.0+" seconds");
			 
			 // Clear the default threshold.
			 model.clearThreshold();
			 model.setThreshold(svmThreshold);
			
			 /*double catThreshold = 0.0;
			 
			 if (tag.equals("beer"))
				 catThreshold = -0.136;
			 if (tag.equals("salad"))
				 catThreshold = -0.132;
			 if (tag.equals("paella"))
				 catThreshold = -0.1345;
			 if (tag.equals("cake"))
				 catThreshold = -0.134;
			 if (tag.equals("steak"))
				 catThreshold = -0.1362;
			 if (tag.equals("coffee"))
				 catThreshold = -0.132;
			 if (tag.equals("hamburger"))
				 catThreshold = -0.13599;
			 if (tag.equals("icecream"))
				 catThreshold = -0.1367;
			 if (tag.equals("pasta"))
				 catThreshold = -0.1353;
			 if (tag.equals("eggs"))
				 catThreshold = -0.137;
			 if (tag.equals("potatoes"))
				 catThreshold = -0.1372;
			 if (tag.equals("pizza"))
				 catThreshold = -0.12805;
			 if (tag.equals("sushi"))
				 catThreshold = -0.1305;
			 if (tag.equals("wine"))
				 catThreshold = -0.1375;
			 
			model.setThreshold(catThreshold);
			 */
			//save SVMModel to disk
			//sVM.saveToDisk(sc.sc(), pathToSVMModel); 
			
			//--------------------- Predict labels for test vectors ----------------------
			 //SVMModel sVMModel = sVM.loadFromDisk(sc.sc(), pathToSVMModel);
		//	 System.out.println("Preparing to classify...");
			 
			 // Create binary testing histograms for this class
			// JavaPairRDD<Double, Vector> testingHists =  SVMForBoW.genClassHistograms(sc, rawTestingHistograms, currLabel);
			// JavaRDD<LabeledPoint> testingData =  testingHists.map(t-> new LabeledPoint(t._1.doubleValue(),t._2));
			
			 // Compute raw scores on the test set.
			 //long timeNow = System.currentTimeMillis();
			 //JavaRDD<Tuple2<Object, Object>> scoreAndLabels = SVMForBoW.predictForData(testingData, model);
			 //double svmPredictTime = (double) (System.currentTimeMillis()- timeNow);
			 //System.out.println("Got the SVM predictions in "+svmPredictTime/1000.0+" seconds");
			 
			 //-------------------  Get evaluation metrics ----------------------------
			 //BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
			 //double auROC = metrics.areaUnderROC();
			 //double auPR = metrics.areaUnderPR();
			 
			
			 //Long trueP = scoreAndLabels.filter( new Function<Tuple2<Object, Object>, Boolean>(){
				// 		private static final long serialVersionUID = 1L;
				 //		@Override
				 //		public Boolean call(final Tuple2<Object, Object> record) {
				 	//		if (record._1.equals(1.0)){
				 		//		if (record._2.equals(1.0))
				 			//		return true;
				 				//else
				 					//return false;}
				 			//else
				 			/*	return false;}
				 		}).count();
					 
			 Long trueN = scoreAndLabels.filter( new Function<Tuple2<Object, Object>, Boolean>(){
				 		private static final long serialVersionUID = 1L;
				 		@Override
				 		public Boolean call(final Tuple2<Object, Object> record) {
				 			if (record._1.equals(0.0)){
				 				if (record._2.equals(0.0))
				 					return true;
				 				else
				 					return false;}
				 			else
				 					return false;}
				 		}).count();
			 
			 Long falseP = scoreAndLabels.filter( new Function<Tuple2<Object, Object>, Boolean>(){
			 		private static final long serialVersionUID = 1L;
			 		@Override
			 		public Boolean call(final Tuple2<Object, Object> record) {
			 			if (record._1.equals(1.0)){
			 				if (record._2.equals(0.0))
			 					return true;
			 				else
			 					return false;}
			 			else
			 				return false;}
			 		}).count();
			 
			 Long falseN = scoreAndLabels.filter( new Function<Tuple2<Object, Object>, Boolean>(){
			 		private static final long serialVersionUID = 1L;
			 		@Override
			 		public Boolean call(final Tuple2<Object, Object> record) {
			 			if (record._1.equals(0.0)){
			 				if (record._2.equals(1.0))
			 					return true;
			 				else
			 					return false;}
			 			else
			 				return false;}
			 		}).count();
			 
			 
			 double precision = trueP.doubleValue()/(trueP.doubleValue()+falseP.doubleValue());
			 double recall = trueP.doubleValue()/(trueP.doubleValue()+falseN.doubleValue());
			 double accuracy = (trueP.doubleValue()+trueN.doubleValue())/(trueP.doubleValue()+trueN.doubleValue()
					 			+falseP.doubleValue()+falseN.doubleValue());
			 double fmeasure = 2*(precision*recall)/(precision+recall);
			 DecimalFormat numberFormat= new DecimalFormat("#0.00");
			 
			 // Print results for this category
			 System.out.println("Category:"+tag+" true positives: "+trueP+" true negatives: "+trueN+" false positives: "+falseP+" false negatives: "+falseN);
			 System.out.println("precision:"+numberFormat.format(precision*100)+"% recall:"+numberFormat.format(recall*100)
					 		+"% accuracy:"+numberFormat.format(accuracy*100)+"% fmeasure:"+numberFormat.format(fmeasure));
			 System.out.println("Area under ROC = " + auROC);
			 System.out.println("Area under PR curve = " + auPR);
			 
		 }

		 long estimatedTime = System.currentTimeMillis() - startTime;
	     long seconds = TimeUnit.MILLISECONDS.toSeconds(estimatedTime);         
	     System.out.println("Execution time: "+seconds+" seconds");
		 sc.close();
	    	*/
	    logwriter.newLine();
	    logwriter.newLine();
	    logwriter.close();
	 } // end of main

} // end of Main Class

	
