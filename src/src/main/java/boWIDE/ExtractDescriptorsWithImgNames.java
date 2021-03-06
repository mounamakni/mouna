package boWIDE;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FilenameUtils;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Size;
//import org.opencv.imgcodecs.Imgcodecs; (opencv 3.0) 
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;   

import scala.Tuple2;
import utils.OpenCVLibrary;

	/**
	 *  * @author Sana Imtiaz
	 * <p><p>
	 * 
	 * The Bag of Words Image Descriptor Extractor class. Creates image descriptors with image IDs
	 */
	public class ExtractDescriptorsWithImgNames implements PairFlatMapFunction<String, String, Vector> {
			  
			private static final long serialVersionUID = 1L;
			
			int resizedImageRows = 0;
			int resizedImageCols = 0;
			int detectorType = 0;
			int extractorType = 0;
			boolean gaussianBlur = false;
			boolean colour = false;
			
			/**
			 * Creates a list of vector descriptors for each image in the form of tuples containing 
			 * imageID and descriptor vector, which is collected as a JavaRDD by the driver program 
			 * <p>
			 * @param resizedImageRows number of rows to resize the image to 
		     * @param resizedImageCols number of columns to resize the image to 
		     * @param detectorType Type of Local descriptor detector (SIFT, SURF, etc.)  
		     * @param extractorType  Type of Local descriptor extractor (SIFT, SURF, etc.)  
		     * @param gaussianBlur Apply Gaussian blur (Default: false)
		     * @param colour Use colour information from images (Default: false)
		     */
			 
			public ExtractDescriptorsWithImgNames(int resizedImageRows, int	resizedImageCols, int detectorType, int extractorType, boolean gaussianBlur, boolean colour) {
		          this.resizedImageRows=resizedImageRows;
		          this.resizedImageCols=resizedImageCols;
		          this.detectorType=detectorType;
		          this.extractorType=extractorType;
		          this.gaussianBlur = gaussianBlur;
		          this.colour = colour;
		    }
			
			
			@Override public Iterable<Tuple2<String,Vector>> call (String s){
				
				// Load runtime OpenCV library on this node
				@SuppressWarnings("unused")
				OpenCVLibrary library = OpenCVLibrary.load();
			
				// Create descriptor detector and extractor
	   		  	FeatureDetector detector = FeatureDetector.create(detectorType);
	   		  	DescriptorExtractor extractor = DescriptorExtractor.create(extractorType);
	   		  	
	   		  	// Read image from disk
	   		  	String imgPath = s;
	   		  	String imgNameWithoutExtension = FilenameUtils.getBaseName(s);
	   		  	Mat imageMat = Highgui.imread(imgPath);	
	   		  		   		  	
	   		  	List<Tuple2<String,Vector>> imageNamesWithDescriptors  = new ArrayList<Tuple2<String,Vector>>();
	   		  	
	   		  	
	   		  	//--------------------------- Pre-processing ------------------------------
	   		  	Mat resizedImage = new Mat (resizedImageRows, resizedImageCols, imageMat.type());		
	   		 
	   		  	if (colour){	
	   		  		// Just Resize image
	   		  		if (imageMat.rows() < resizedImage.rows() || imageMat.cols() < resizedImage.cols())
	   		  			Imgproc.resize(imageMat, resizedImage, resizedImage.size(), 0, 0, Imgproc.INTER_LINEAR);
	   		  		else 
	   		  			Imgproc.resize(imageMat, resizedImage, resizedImage.size(), 0, 0, Imgproc.INTER_CUBIC); }
	   		  	else{
	   		  		// Convert to grayscale
		 			Mat grayImage = new Mat(imageMat.rows(), imageMat.cols(), imageMat.type());
		 			Imgproc.cvtColor(imageMat, grayImage, Imgproc.COLOR_BGRA2GRAY);
		 			Core.normalize(grayImage, grayImage, 0, 255, Core.NORM_MINMAX);
		 			
		 			// Now Resize image
		 			if (imageMat.rows() < resizedImage.rows() || imageMat.cols() < resizedImage.cols())
		 				Imgproc.resize(grayImage, resizedImage, resizedImage.size(), 0, 0, Imgproc.INTER_LINEAR);
		 			else 
		 				Imgproc.resize(grayImage, resizedImage, resizedImage.size(), 0, 0, Imgproc.INTER_CUBIC);
		 		}
	
	   		  	// Apply Gaussian blur if specified
	   		  //	Mat blurredImage = new Mat (resizedImageRows, resizedImageCols, imageMat.type());
	   		  //	if (gaussianBlur){
	   		  //		int gaussianFilterSize = 5;
	   		  //		Imgproc.GaussianBlur(resizedImage, blurredImage,new Size(gaussianFilterSize,gaussianFilterSize), 2.2, 2);
	   		  //	}
	   		  //	else
	   		  //		blurredImage = resizedImage.clone();
	   		  	
	   		  	// ----------------- Detecting the features and creating descriptor vectors ----------------- 
	   		  	MatOfKeyPoint keyPoints = new MatOfKeyPoint();
	   		  	detector.detect(resizedImage, keyPoints);//blurred
			
	   		  	// Compute descriptors based on key-points
	   		  	Mat descriptors = new Mat();
	   		  	extractor.compute(resizedImage, keyPoints, descriptors);//blurred
		
	   		  	//----------------- Convert Matrix of descriptors to Spark Vectors ------------------------				
	   		  	int descriptorRows = descriptors.rows();
	   		  	Mat descriptorsDouble = new Mat();
	   		    
	   		  	// Convert the descriptors from float (Default) to Double format
	   		  	descriptors.convertTo(descriptorsDouble, CvType.CV_64FC1);
			
	   		  	for (int idx= 0; idx <descriptorRows ; idx++){
	   		  		double[] doubleData = new double [descriptorsDouble.cols()];
	   		  		
	   		  		// Read an image descriptor
	   		  		descriptorsDouble.get(idx,0,doubleData);
	   		  		
	   		  		// Create a Spark Vector for this descriptor
	   		  		Vector dv = Vectors.dense(doubleData);
	   		  		
	   		  		// Create a Tuple with imageID and descriptor vector, and add it to the list
	   		  		Tuple2<String,Vector> descriptor = new Tuple2<String, Vector> (imgNameWithoutExtension,dv);
	   		  		imageNamesWithDescriptors.add(descriptor);
	   		  	}
	        	
	   		  	return imageNamesWithDescriptors;}
	
	}   // end of Class
		

