package hageldave.optisled.generic.solver;

import java.util.HashMap;

public class HyperParams {
	protected HashMap<String, Object> paramMap = new HashMap<>();
	
	@SuppressWarnings("unchecked")
	public <T> T get(String paramName) {
		return (T) this.paramMap.get(paramName);
	}
	
	@SuppressWarnings("unchecked")
	public <T> T getOrDefault(String paramName, T defaultVal) {
		return has(paramName) ? (T) this.paramMap.get(paramName) : defaultVal;
	}
	
	public <T> void set(String paramName, T value) {
		this.paramMap.put(paramName, value);
	}
	
	public boolean has(String paramName) {
		return this.paramMap.containsKey(paramName);
	}
	
}
