package boWModel;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.rdd.RDD;
import org.apache.spark.SparkContext;

/**
 * @author Sana Imtiaz
 * <p><p>
 * Class containing the supporting functions for Bag of Words Model, <b>BoWModel</b>
 */
public class BoWModel {
	
	private static BoWModel boWModelSingleton = null;

    /**
     * Gets a single instance of BoWModel
     * <p><p>
     * @return a BoWModel
     */
    public static BoWModel getSingleton() {
        
    	if (boWModelSingleton == null) {
            boWModelSingleton = new BoWModel();
            }
        return boWModelSingleton;
    } 

    
    
    /**
     * Creates and trains a k-means model for a set of input image descriptors
     * <p>
     * @param vectorDescriptors image descriptors to be clustered
     * @param numClusters the <b>k</b> - number of clusters to create
     * @param numIterations number of maximum iterations for the algorithm 
     * <p>
     * @return KMeansModel for the input data
     */
    public KMeansModel createModel (JavaRDD<Vector> vectorDescriptors, int numClusters, int numIterations){

    	KMeansModel clusters = KMeans.train(vectorDescriptors.rdd(), numClusters, numIterations);
		return clusters;
		
	}

    
	
	/**
	 * Creates and trains a k-means model for a set of input image descriptors
     * <p>
     * @param vectorDescriptors image descriptors to be clustered
     * @param numClusters the <b>k</b> - number of clusters to create
     * @param numIterations number of maximum iterations for the algorithm 
     * @param parallelRuns number of iterations of algorithm to be run in parallel (Default: 1)
	 * <p>
	 * @return KMeansModel for the input data
	 */
	public KMeansModel createModel (JavaRDD<Vector> vectorDescriptors, int numClusters, int numIterations, int numParallelRuns){
		
		KMeansModel clusters = KMeans.train(vectorDescriptors.rdd(), numClusters, numIterations, numParallelRuns);
		return clusters;
		
	} 
	
	
    /**
     * Creates and trains a k-means model for a set of input image descriptors
     * <p>
     * @param vectorDescriptors image descriptors to be clustered
     * @param numClusters the <b>k</b> - number of clusters to create
     * @param numIterations number of maximum iterations for the algorithm 
     * @param parallelRuns number of iterations of algorithm to be run in parallel (Default: 1)
     * @param initMode initialization model, either "random" or "k-means||" (Default).
     * <p>
     * @return KMeansModel for the input data
     */
	public KMeansModel createModel (JavaRDD<Vector> vectorDescriptors, int numClusters, int numIterations, int parallelRuns, String initMode){

		KMeansModel clusters = KMeans.train(vectorDescriptors.rdd(), numClusters, numIterations, parallelRuns, initMode);
		return clusters;
		
	}


	/**
	 * Returns the k-means cost (sum of squared distances of points to their nearest center) for this model on the given data.
	 * <p>
	 * @param clusters the k-Means model
	 * @param vectorDescriptors input data to compute the cost
	 * <p>
	 * @return  k-means cost (The lower the cost, the better the clustering) 
	 */
	public double computeWSSSE (KMeansModel clusters, RDD<Vector> vectorDescriptors){
	
		double WSSSE = clusters.computeCost(vectorDescriptors);
		return WSSSE;
		
    } 
	
	
	
	/**
	 * Loads k-Means model from the disk
	 * <p> 
	 * @param sc the Spark Context
	 * @param pathToKmeansModel path to k-means model on disk
	 * <p>
	 * @return KMeansModel
	 * 
	 */
	public KMeansModel loadFromDisk (SparkContext sc, String pathToKmeansModel){
		KMeansModel clusters = KMeansModel.load(sc, pathToKmeansModel);
		return clusters;
		
	}
    
	
	/**
	 * Saves k-Means model to the disk
	 * <p>
	 * @param clusters the k-means model
	 * @param sc the Spark Context
	 * @param pathToKmeansModel path to k-means model on disk
	 * <p>
	 * @throws IOException
	 */
	public void saveToDisk (KMeansModel clusters, SparkContext sc, String pathToKmeansModel) throws IOException{
    	
		// Clean the directory if it exists
		FileUtils.cleanDirectory(new File(pathToKmeansModel)); 
		clusters.save(sc, pathToKmeansModel);
		
	}
    	

} // end of Class

	