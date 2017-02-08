package sVM;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.io.Serializable;

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

/**
 * @author Sana Imtiaz
 * <p><p>
 * Class containing the supporting functions for Support Vector Machine (SVM) for the Bag of Words Model
 */

public class SVMForBoW implements Serializable{
	
	private static final long serialVersionUID = 1L;
	
	/**
	 * Predicts the labels for input Data
	 * <p>
	 * @param testData JavaRDD of LabeledPoints containing testing Histograms
	 * @param model the SVM model
	 * <p>
	 * @return JavaRDD containing List of Tuples in the format "predictedLabel, originalLabel"
	 */
	public static JavaRDD<Tuple2<Object, Object>> predictForData(JavaRDD<LabeledPoint> testData, SVMModel model){
		
		JavaRDD<Tuple2<Object, Object>> scoreAndLabels = testData.map(new 
				Function<LabeledPoint, Tuple2<Object, Object>>(){ 				
					private static final long serialVersionUID = 1L;
					
					// For each histogram <LabeledPoint.features()>, predict label <score> using SVM model and create Tuple <score, LabeledPoint.label()> 
					public Tuple2<Object, Object> call(LabeledPoint p) {
						Double score = model.predict(p.features());
						return new Tuple2<Object, Object>(score, p.label());}
	    		}
			);
		return scoreAndLabels;
	}
	
    
	/**
	 * Generate binary-class histograms for the specified class provided multiclass histograms
	 * <p>
	 * @param sc the Spark Context
	 * @param originalHists the Multiclass histograms
	 * @param classLabel numeric label for the positively-marked class
	 * <p>
	 * @return Binary-Class Histograms for specified classLabel 
	 */
	public static JavaPairRDD<Double,Vector> genClassHistograms (JavaSparkContext sc, JavaPairRDD<Double, Vector> originalHists, Double classLabel){
		List<Tuple2<Double,Vector>> classHistogramsList = new ArrayList<Tuple2<Double, Vector>>();
		List<Tuple2<Double,Vector>> originalHistograms = originalHists.collect(); 
		for (Tuple2<Double, Vector> t: originalHistograms){
			if (t._1.equals(classLabel))
				classHistogramsList.add(new Tuple2<Double, Vector> (new Double(1.0),t._2));
			else 
				classHistogramsList.add(new Tuple2<Double, Vector> (new Double(0.0),t._2));
		}
		
		JavaPairRDD<Double,Vector> classHistograms = sc.parallelizePairs(classHistogramsList);
		return classHistograms;
	} 
	
	
	
	/**
	 * Load SVM model from disk
	 * <p>
	 * @param sc the Spark Context
	 * @param pathToSVMModel path to SVM model on disk
	 * <p>
	 * @return SVM model
	 * @throws IOException
	 */
	public static SVMModel loadSVMFromDisk (SparkContext sc, String pathToSVMModel)throws IOException{
		SVMModel sVMModel = SVMModel.load(sc, pathToSVMModel);
		return sVMModel;
	}
	
	
	/**
	 * Save SVM model to disk
	 * <p>
	 * @param sc the Spark Context
	 * @param pathToSVMModel  path to directory for SVM model on disk
	 * @param sVMModel the SVM model
	 * <p>
	 * @throws IOException
	 */
	public static void saveSVMToDisk (SparkContext sc, String pathToSVMModel, SVMModel sVMModel)throws IOException{
		File sVMDir = new File(pathToSVMModel);
		FileUtils.cleanDirectory(sVMDir);
		FileUtils.deleteDirectory(sVMDir);
		
		sVMModel.save(sc, pathToSVMModel);
	}
	
	
} //  end of Class
