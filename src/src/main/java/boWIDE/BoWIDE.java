package boWIDE;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.SparkContext;

import java.io.Serializable;

	/**
	 *  * @author Sana Imtiaz
	 * <p><p>
	 * 
	 * The Bag of Words Image Descriptor Extractor class
	 */
	public class BoWIDE implements Serializable{
		
		private static final long serialVersionUID = 1L;
	
		private static BoWIDE boWIDESingleton = null;
		
		public int clusters = 0;
		
		/**
		 * Get a singleton object of this Class
		 * @return BoWIDE object
		 */
		public static BoWIDE getSingleton(){
	
			if (boWIDESingleton == null) {
				boWIDESingleton = new BoWIDE();
	            }
	        return boWIDESingleton;
		}
		
		
		/**
		 * Extracts the Bag-of-Words Bags (Histograms) for the provided images
		 * <p>
		 * @param imageNamesWithDescriptors JavaPairRDD containing images with respective descriptors
		 * @param dictionary the k-means model - Bag of words
		 * <p>
		 * @return BoW Histograms (Bags) for input images
		 */
		public JavaPairRDD<String,Vector> getBoWBags (JavaPairRDD<String,Vector> imageNamesWithDescriptors, KMeansModel dictionary){
			
			// Get the number of clusters in dictionary (histogram size)
			clusters = dictionary.k();
			
			// Map the image descriptors to the best matching BoW dictionary cluster
			JavaPairRDD<String,Integer> imageClusterMappings = imageNamesWithDescriptors.mapValues(x -> dictionary.predict(x));
			
			// Collect mappings list for each Image separately
			JavaPairRDD<String,Iterable<Integer>> imageClusterPoints = imageClusterMappings.groupByKey();
						
			// Count mapped cluster point instances for each Image to create histogram
			JavaPairRDD<String,double[]> histograms = imageClusterPoints.mapValues(new Function<Iterable<Integer>, double[]>(){
				private static final long serialVersionUID = 1L;
	
				@Override public double[] call (Iterable<Integer> clusterPoints){
					int totalDesciptors = 0;
					double[] histogram = new double[clusters];
					int[] histogramCounts = new int[clusters];
					
					for (Integer i:clusterPoints)
						histogramCounts [i] = histogramCounts[i]+1; 		
					for (int count=0; count<clusters; count++)
						totalDesciptors += histogramCounts[count];
					for (int idx=0; idx<clusters; idx++)
						histogram[idx] = histogramCounts[idx]/(double)totalDesciptors;
					
					return histogram;
					}
			}
			);
			
			// Create a JavaPairRDD containing image paths and their respective histograms
			JavaPairRDD<String, Vector> sVMHistograms = histograms.mapValues(x -> Vectors.dense(x));
			return sVMHistograms;
			
		}
		
	
		/**
		 * Loads the k-means dictionary from disk
		 * <p>
		 * @param sc the Spark Context
		 * @param path path to k-means dictionary on disk
		 * <p>
		 * @return KMeansModel for the bag of words dictionary
		 */
		public KMeansModel loadDictionary (SparkContext sc, String path){
		
			KMeansModel dictionary = KMeansModel.load(sc, path);
			clusters = dictionary.k();
			System.out.println("Dictionary loaded from: "+path);
			return dictionary;	
		}
	
	
	
	} // end of Class
	
		
