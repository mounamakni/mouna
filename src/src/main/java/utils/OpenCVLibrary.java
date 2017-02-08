package utils;

import org.opencv.core.Core;

/**
 * @author Sana Imtiaz
 * <p>
 * Class with singleton function to load runtime OpenCV library
 */
public class OpenCVLibrary {
	
	private static OpenCVLibrary library = null;

	/**
	 * Loads runtime OpenCV library on the calling node, when called for the first time
	 * <p>
	 * @return loaded runtime OpenCV library
	 */
	 public static OpenCVLibrary load() {
	        
	    	if (library == null) {
	    		library = new OpenCVLibrary();
	    		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	        }
	    	return library;
	    }
}
