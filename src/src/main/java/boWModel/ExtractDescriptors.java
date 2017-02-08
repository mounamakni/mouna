package boWModel;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.function.FlatMapFunction;
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

import utils.OpenCVLibrary;

	/**
	  @author Sana Imtiaz
	 * <p><p>
	 * Customized FlatMap function for extracting local descriptors from specified images using OpenCV and translating them to Spark RDD.
	 * 
	 */
	public class ExtractDescriptors implements FlatMapFunction<String, Vector> {
	  
		private static final long serialVersionUID = 1L;
		int resizedImageRows;
	    int resizedImageCols;
	    int detectorType;
	    int extractorType;
	    boolean gaussianBlur;
	    boolean colour;
	    
	    /**
	     * Creates a list of vector descriptors for each image,
	     * which is collected as a JavaRDD by the driver program 
	     * <p>
	     * @param resizedImageRows number of rows to resize the image to 
	     * @param resizedImageCols number of columns to resize the image to 
	     * @param detectorType Type of Local descriptor detector (SIFT, SURF, etc.)  
	     * @param extractorType  Type of Local descriptor extractor (SIFT, SURF, etc.)  
	     * @param gaussianBlur Apply Gaussian blur (Default: false)
	     * @param colour Use colour information from images (Default: false)
	     */
	    public ExtractDescriptors(int resizedImageRows, int	resizedImageCols, int detectorType, int extractorType, boolean gaussianBlur, boolean colour) {
	          this.resizedImageRows=resizedImageRows;
	          this.resizedImageCols=resizedImageCols;
	          this.detectorType=detectorType;
	          this.extractorType=extractorType;
	          this.gaussianBlur = gaussianBlur;
	          this.colour = colour;
	    }
		
	
	@Override public Iterable<Vector> call (String s){
		String pathToLog = "/gpfs/projects/bsc31/bsc31914/tests/data/logextract.txt";//"/Users/mouna/these/bow-master/SparkVersion/data/logextract.txt";
		//BufferedWriter logwriter = null;
		/*try {
			logwriter = new BufferedWriter(new FileWriter(pathToLog, true));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}//append
		*/
		// Load runtime OpenCV library on this node
		@SuppressWarnings("unused")
		OpenCVLibrary library = OpenCVLibrary.load();
	    
		// Create descriptor detector and extractor
	    FeatureDetector detector = FeatureDetector.create(detectorType);
	    DescriptorExtractor extractor = DescriptorExtractor.create(extractorType);	
	    
		// Read image from disk
	    String imgPath = s;
		Mat imageMat = 	Highgui.imread(imgPath);	
	/*	try {
			logwriter.newLine();
			logwriter.write("Image: "+imgPath);
			logwriter.newLine();
			//logwriter.write("Mat rows: "+imageMat.rows() +"   Mat height: "+ imageMat.height() );
			//logwriter.newLine();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		*/
		List<Vector> descriptorsVectors;
		
		//--------------------------- Pre-processing ------------------------------
		Mat resizedImage = new Mat (resizedImageRows, resizedImageCols, imageMat.type());		
		if (colour){	
	  		// Just Resize image
	  		if (imageMat.rows() < resizedImage.rows() || imageMat.cols() < resizedImage.cols())
	  			Imgproc.resize(imageMat, resizedImage, resizedImage.size(), 0, 0, Imgproc.INTER_LINEAR);
	  		else 
	  			Imgproc.resize(imageMat, resizedImage, resizedImage.size(), 0, 0, Imgproc.INTER_CUBIC);
			}
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
		/*Mat blurredImage = new Mat (resizedImageRows, resizedImageCols, imageMat.type());
		if (gaussianBlur){
			int gaussianFilterSize = 5;
			Imgproc.GaussianBlur(resizedImage, blurredImage,new Size(gaussianFilterSize,gaussianFilterSize), 2.2, 2);
		}
		else
			blurredImage = resizedImage.clone();
		*/
		// ----------------- Detecting the features and creating descriptor vectors ----------------- 
		MatOfKeyPoint keyPoints = new MatOfKeyPoint();
		detector.detect(resizedImage, keyPoints);//resized image instead of blurred
		
		// Compute descriptors based on key-points
		Mat descriptors = new Mat();
		extractor.compute(resizedImage, keyPoints, descriptors);//resized image instead of blurred
	/*	try {
			
			logwriter.write("Image has: "+keyPoints.total() + " keypoints");
			logwriter.newLine();
			logwriter.write("Descriptors size " + descriptors.size() );//"+ " descriptors / rows = " + descriptors.rows() + " / cols = "+descriptors.cols());
			logwriter.newLine();
			/*for (int i=0;i<384;i++)
			{logwriter.write(descriptors.get(0, i)+", ");}
			logwriter.newLine();*/
			//descriptors.
		/*	

		} catch (IOException e) {
			e.printStackTrace();
		}*/
		//----------------- Convert Matrix of descriptors to Spark Vectors ------------------------		
		int descriptorRows = descriptors.rows();
		//System.out.println("Image:"+imgPath+"\t descriptors:"+descriptorRows);
		Mat descriptorsDouble = new Mat();
		descriptorsVectors =new ArrayList<Vector>();
		// Convert the descriptors from float (Default) to Double format
		descriptors.convertTo(descriptorsDouble, CvType.CV_64FC1);
		
		for (int idx= 0; idx <descriptorRows ; idx++){
			double[] doubleData = new double [descriptorsDouble.cols()];
			
			// Read an image descriptor
			descriptorsDouble.get(idx,0,doubleData);
			
			//try {
				//logwriter.write("doubledata lengh: "+doubleData.length);
				//logwriter.write(doubleData[0] + ";"+doubleData[1] + ";"+doubleData[2] + ";"+doubleData[3] + ";");
				//logwriter.newLine();
				
			//} catch (IOException e) {
			//	e.printStackTrace();
			//}
				
			// Create a Spark Vector for this descriptor
			Vector dv = Vectors.dense(doubleData);
			
			// Add to list of images' descriptors
			descriptorsVectors.add(dv);
		}
		
			
			//for (int i = 0; i < descriptorsVectors.size(); i++) {
			/*   
				try {
					//logwriter.write("Size descriptors vector "+ descriptorsVectors.size());//descriptorsVectors.get(0).toString());
					//logwriter.newLine();
					logwriter.close();
				}catch (IOException e) {
					e.printStackTrace();
				}
		 */
		
		return descriptorsVectors; // the images' descriptors
		}

}    // end of Class


