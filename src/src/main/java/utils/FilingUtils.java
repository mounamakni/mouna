package utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Map;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

/**
 * @author Sana Imtiaz. 
 * 
 * 	<p><p>
 * 	<b> Class for Filing Utilities. </b>
 *  <p>	Contains functions for writing input data paths to files, writing BOW Histograms to file and for 
 *	writing/reading the SVM Category-to-Label map to/from disk 
 *	
 */
public class FilingUtils {
	
	
	/**
	 * Writes paths to training and testing data (images and metadata) in separate files, so Spark can directly load them into separate RDDs
	 *  
	 *  <p><p>
	 * @param inputImagesPath 			Path to the directory containing all images 
	 * @param trainingImagesFilePath 	File for storing paths to training images
	 * @param testingImagesFilePath 	File for storing paths to testing images
	 * @param inputMetadataPath 		Path to the directory containing all metadata
	 * @param trainingMetadataFilePath 	File for storing paths to metadata for training images
	 * @param testingMetadataFilePath 	File for storing paths to metadata for testing images
	 * @param trainingFraction 			Fraction for train/test sets distribution.  (For example, 
	 * 0.8 denotes 80% of dataset to be  used for training set and 20% for testing)
	 * <p>
	 * @throws IOException
	 */
	public static void writePathsToFiles (String inputImagesPath, String trainingImagesFilePath, String testingImagesFilePath, String inputMetadataPath,
				String trainingMetadataFilePath, String testingMetadataFilePath, double trainingFraction) throws IOException{
		
		
		//  Read the contents of Input directory in a String Array
		File inputDir = new File(inputImagesPath);
		String[] inputDirList = inputDir.list();
		System.out.println("TAILLLLLLEEEE LISTE   "+inputDirList.length);
		
		 // Calculate number of files for training set
		int maxFilesToProcess = (int) (inputDirList.length * trainingFraction);
		int filesProcessed = 0;
		
		// Create separate buffered writer for each file
		BufferedWriter bwImTrain = new BufferedWriter(new FileWriter(trainingImagesFilePath));
		BufferedWriter bwImTest = new BufferedWriter(new FileWriter(testingImagesFilePath));
		BufferedWriter bwMTrain = new BufferedWriter(new FileWriter(trainingMetadataFilePath));
		BufferedWriter bwMTest = new BufferedWriter(new FileWriter(testingMetadataFilePath));
		
		// Iterate through images' directory and write paths to training images and metadata in the respective files
		while (filesProcessed <  maxFilesToProcess)
		{
			String imgPath = inputDirList[filesProcessed];
			//1-comment System.out.println("FILE "+inputImagesPath+File.separator+imgPath);
			//if(imgPath.contains("jpg"))
			//{
			bwImTrain.write(inputImagesPath+File.separator+imgPath);
			bwImTrain.newLine();
			
			String metadatapath = imgPath.replace(".jpg", ".json");
			bwMTrain.write(inputMetadataPath+File.separator+metadatapath);
			bwMTrain.newLine();
			
			filesProcessed++;
			//}
		}
		bwImTrain.close();
		bwMTrain.close();
		
		// The rest of images and metadata belongs to testing set, write paths in the respective files
		while (filesProcessed < inputDirList.length)
		{
			String imgPath = inputDirList[filesProcessed];
			bwImTest.write(inputImagesPath+File.separator+imgPath);
			bwImTest.newLine();
			
			String metadatapath = imgPath.replace(".jpg", ".json");
			bwMTest.write(inputMetadataPath+File.separator+metadatapath);
			bwMTest.newLine();
			
			filesProcessed++;
		}
		bwImTest.close();
		bwMTest.close();
			
	} // end of function body
	
	
	/**
	 * Saves BOW Histograms to File on disk in the format <b>"label 1:val1 2:val2 3:val3 ... N:valN"</b>
	 * 
	 * <p><p>
	 * @param rawTrainingHistograms JavaRDD containing multiclass training histograms
	 * @param pathToSVMTrainingData path to file on disk
	 * <p>
	 * @throws IOException
	 */
	public static void saveHistogramsToFile (JavaPairRDD<Double, Vector> rawTrainingHistograms, String pathToSVMTrainingData) throws IOException{
		
		//	Set format to store histogram values up to 5 decimal places
		DecimalFormat numberFormat= new DecimalFormat("#0.00000");
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(pathToSVMTrainingData)); 
		//	Read all the Histograms and labels from JavaPairRDD into a list
		List<Tuple2<Double, Vector>> rawHistograms = rawTrainingHistograms.collect();
		
		//  For each histogram, create a new line in file and read values from the Tuple. 
		//	Store as "label 1:val1 2:val2 3:val3 ... N:valN" 
		for (Tuple2<Double, Vector> t: rawHistograms){
	    	String histValsString = "";
	    	Double currLabel = t._1;	// the multiclass label
	    	Vector histVals = t._2;		// BOW histogram
	    	double[] histValsDouble = histVals.toArray();
	    	for (int idx=0; idx<histValsDouble.length; idx++){
	    		 histValsString = histValsString+(idx+1)+":"+numberFormat.format(histValsDouble[idx])+" ";
	    	}
	    	bw.write(currLabel.doubleValue()+" "+histValsString);
	    	bw.newLine();
	    }
		bw.close();
		
	} // end of function body
	
	
	
	/**
	 * Loads the category to numeric labels map from disk in the form of <b>Map<String, Double></b> 
	 * 
	 * <p><p>
	 * @param labelsPath	Path to file containing the category-to-label map
	 * @return				Category-to-label map as Map<String, Double> 
	 * <p> 
	 * @throws Exception
	 */
	@SuppressWarnings("unchecked")
	public static Map<String, Double> loadLabelsFromDisk(String labelsPath) throws Exception{
	    	ObjectInputStream ois = new ObjectInputStream(new FileInputStream(labelsPath));
	        Object result = ois.readObject();
	        Map<String, Double> labels = (Map<String, Double>)result;
	        ois.close();
	        return labels;
	} 	// end of function body
	
	
	
	/**
	 * Saves the Category-to-label map to disk
	 * 
	 * <p><p>
	 * @param labelsPath	Path to file for storing the category-to-label map
	 * @param categoryLabels	Category-to-label map in the form Map<String, Double>  
	 * <p>
	 * @throws Exception
	 */
	public static void saveLabelsToDisk(String labelsPath,  Map<String, Double> categoryLabels) throws Exception{
	    	FileOutputStream fos = new FileOutputStream(labelsPath);   
	    	ObjectOutputStream oos = new ObjectOutputStream(fos);           
	    	oos.writeObject(categoryLabels); 
	    	oos.close();
	}	// end of function body
	    
	    
	
}	// end of class
