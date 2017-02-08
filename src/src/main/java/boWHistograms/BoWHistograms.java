package boWHistograms;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

/**
 *  * @author Sana Imtiaz
 * <p><p>
 * 
 * The Bag of Words Histograms class. Takes the multiclass histograms with name labels and transforms it to SVM histogram format
 */

public class BoWHistograms {
	
	public static Map<String,Double> labels;	// Java Map containing category names and assigned numeric labels
	
	
	/**
	 * Creates a JavaPairRDD containing image IDs and respective tags indicated from metadata files
	 * <p>
	 * @param metadataPaths RDD containing paths to metadata files
	 * <p>
	 * @return JavaPairRDD containing image IDs with respective tags
	 */
	public static JavaPairRDD<String,String> getTags(JavaRDD<String> metadataPaths){
		JavaPairRDD<String,String> idsWithMetadata = metadataPaths.flatMapToPair(new GetTags());
		return idsWithMetadata;
	}
	
	
	/**
	 * Creates multiclass training histograms for images in the format (classLabel,histogram)
	 * <p>
	 * @param idsWithHistograms JavaPairRDD containing image IDs with respective training vectors (histograms)
	 * @param idsWithTags JavaPairRDD containing image IDs with respective category tags (as indicated by metadata)
	 * @param categoryLabels Map containing numeric labels for each category name
	 * <p>
	 * @return JavaPairRDD of multiclass training histograms
	 */
	public static JavaPairRDD<Double,Vector> getRawHistograms(JavaPairRDD<String,Vector> idsWithHistograms, JavaPairRDD<String, String> idsWithTags, Map<String, Double> categoryLabels){
		
		// Replace category tags with respective numeric labels
		JavaPairRDD<String, Double> idsWithLabels = idsWithTags.mapValues(x-> categoryLabels.get(x));
		System.out.println("Creating "+idsWithLabels.count()+" histograms for the classifier...");
		
		// Join the image IDs and respective numeric labels with histograms
		JavaPairRDD<String, Tuple2<Double, Vector>> idsWithHistAndTags = idsWithLabels.join(idsWithHistograms);
		
		// Extract the training vectors (classLabel,histogram) 
		JavaRDD<Tuple2<Double, Vector>>  rawIDsHistAndTags = idsWithHistAndTags.values();
		JavaPairRDD<Double,Vector> rawHistogramsAndTags = JavaPairRDD.fromJavaRDD(rawIDsHistAndTags);
		return rawHistogramsAndTags;
	}
	
	
	
	/**
	 * Gets numeric label for specified category name
	 * <p>
	 * @param categoryName category Name
	 * <p>
	 * @return numeric label for this category
	 */
	public static Double getLabelForTag (String categoryName){
		Double label = labels.get(categoryName);
		return label;
	}
	
	
	/**
	 * Gets the Java Map containing category names and numeric labels
	 * <p>
	 * @return  Map containing category names and assigned numeric labels
	 */
	public static Map<String,Double> getTagsToLabelMap(){
		return labels;
	}
		
	
	/**
	 * Creates map with all category names and assigns numeric labels
	 * <p>
	 * @param idsWithTags image IDs with category names as provided by metadata files
	 * <p>
	 * @return Map containing category names and assigned numeric labels
	 */
	
	public static Map<String,Double> createLabelsFromTags (JavaPairRDD<String, String> idsWithTags){
		Map<String, Double> categoryLabels = new HashMap<String,Double>();
		
		// Extract category names as specified by tags for this collection of images
		List<String> categoryNames = idsWithTags.values().distinct().collect();
		Double count = 0.0;
		
		// Assign a numeric label to each category and add it to map
		for (String category: categoryNames){
			categoryLabels.put(category, count);
			count++;
		}
		labels = categoryLabels;
		
		return categoryLabels;
	}
	

} // end of class
