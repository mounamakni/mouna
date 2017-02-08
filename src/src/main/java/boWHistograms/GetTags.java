package boWHistograms;

import java.util.ArrayList;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FilenameUtils;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import scala.Tuple2;

/**
 * @author Sana Imtiaz
 * <p><p>
 * 
 * Class containing customized flatmap function for pairs which creates JavaPairRDD 
 * containing images IDs with respective category tags as specified in metadata files.
 * 
 *  Input parameters: paths to metadata files for images
 *  Returns: JavaPairRDD with image IDs and tags
 */


public class GetTags implements PairFlatMapFunction<String, String, String> {
	 
		private static final long serialVersionUID = 1L;
		
	@Override public Iterable<Tuple2<String,String>> call (String paths) throws FileNotFoundException, IOException, ParseException{
		List<Tuple2<String, String>> imageIdsWithTags = new ArrayList<Tuple2<String,String>>();
		
		// Read metadata (JSON format) in a JSON object
		JSONParser parser = new JSONParser();
		Object obj = parser.parse(new FileReader(paths));
		JSONObject jsonObject = (JSONObject) obj;
        
		// Extract image category tag specified in user-provided "tags"
		JSONArray tags = (JSONArray) jsonObject.get("tags");
        String tag = (String) tags.get(0);
		
        // Create a Tuple with imageID and extracted tag. Add it to the list!
        Tuple2<String,String> imageTagTuple = new Tuple2<String,String>(FilenameUtils.getBaseName(paths),tag);
        imageIdsWithTags.add(imageTagTuple);
		
		return imageIdsWithTags;
	}
}