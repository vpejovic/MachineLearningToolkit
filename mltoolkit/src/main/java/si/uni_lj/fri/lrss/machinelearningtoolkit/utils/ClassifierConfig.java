package si.uni_lj.fri.lrss.machinelearningtoolkit.utils;

import java.util.HashMap;
import java.util.Set;

public class ClassifierConfig {

	private HashMap<String, Object> mParams;
		
	public ClassifierConfig() {
		
		mParams = new HashMap<String, Object>();
		
	}
	
	public void addParam(String param, Object value){
		mParams.put(param, value);
	}
	
	public Object getParam(String param) {
		if (mParams.containsKey(param)) {
			return mParams.get(param);
		}
		return null;
	}
	
	public boolean containsParam(String param) {
		
		return mParams.containsKey(param);
	}
	
	public Set<String> getAllParams() {
		return mParams.keySet();
	}
}
