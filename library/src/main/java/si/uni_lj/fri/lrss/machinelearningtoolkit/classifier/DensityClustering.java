/*
 *  Copyright (c) 2013, University of Birmingham, UK,
 *  Copyright (c) 2013, University of Ljubljana, Slovenia,
 *  Veljko Pejovic,  <Veljko.Pejovic@fri.uni-lj.si>
 *
 *
 *  This library was developed as part of the EPSRC Ubhave (Ubiquitous and Social
 *  Computing for Positive Behaviour Change) Project. For more information, please visit
 *  http://www.ubhave.org
 *
 *  Permission to use, copy, modify, and/or distribute this software for any purpose with
 *  or without fee is hereby granted, provided that the above copyright notice and this
 *  permission notice appear in all copies.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 *  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 *  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 *  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 *  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 *  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */
package si.uni_lj.fri.lrss.machinelearningtoolkit.classifier;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import android.util.Log;

import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.ClassifierConfig;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Constants;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Feature;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.FeatureNominal;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Instance;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.MLException;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Signature;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Value;

/**
 * This classifier calculates centroids of labelled, clustered data instances.
 * The classifier first removes cluster outliers using the density method,
 * i.e. if less than a given percentage of other data instances are in the 
 * epsilon environment of a point, the point is removed as an outlier. 
 * Cluster centroids are then calculated. The classifier is not an online 
 * classifier, and is batch trained from given instances. 
 * 
 * At the classification time, an instance is given a label that corresponds
 * to the closest cluster centroid. The distance used for density and closeness
 * calculation is euclidean distance. In case only two nominal features exist, 
 * the classifier assumes that GPS coordinates are given, thus it adjusts the 
 * distance calculation according to the harvesine formula.
 * 
 * The classifier can only be instantiated with a nominal class feature and one 
 * or more numeric attribute features. 
 * 
 * LIMITATION: If no data are present for a label, the assigned centroid will 
 * have all zeros coordinates.
 * 
 * @author Veljko Pejovic, University of Birmingham, UK <v.pejovic@cs.bham.ac.uk>
 *
 */
public class DensityClustering extends Classifier {

	private static final String TAG = "DensityClustering";
	
	private HashMap<String,double[]> mCentroids;
	
	private HashMap<String,Integer> mNumTrains;
	
	private double mMaxDistance;
	
	private double mMinInclusionPct;
	
	private static double toRad(double a_degree) {
		return Math.PI*a_degree/180.0;
	}
	
	private static double distance(final double[] coordsA,final double[] coordsB)
			throws MLException {
		
		if (coordsA.length != coordsB.length) {
			throw new MLException(MLException.INCOMPATIBLE_INSTANCE, 
					"Instance is not compatible with the dataset used for classifier construction.");					
		}
		
		// We assume GPS coordinates if vectors of size two are given
		if (coordsA.length == 2) {
			
			double lat1 = coordsA[0];
			double lon1 = coordsA[1];
			double lat2 = coordsB[0];
			double lon2 = coordsB[1];
			double R = 6371.0;
			double dLat = toRad(lat2 - lat1);
			double dLon = toRad(lon2 - lon1);
			double radLat1 = toRad(lat1);
			double radLat2 = toRad(lat2);
			
			double a = Math.sin(dLat/2.0) * Math.sin(dLat/2.0) + 
					Math.sin(dLon/2.0) * Math.sin(dLon/2.0) * Math.cos(radLat1) * Math.cos(radLat2);
			double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
			
			return R * c * 1000.0;		
		}
		// Otherwise Euclidean distance
		else {
			double sqrSum = 0;			
			for(int i=0; i<coordsA.length; i++) {
				sqrSum += Math.pow(coordsA[i] - coordsB[i], 2);
			}
			return Math.sqrt(sqrSum);
		}		
	}

    /**
     * Creates a new density clustering classifier.
     * @param signature Signature of the classifier.
     * @param config Optional configuration parameters.
     */
	public DensityClustering(Signature signature, ClassifierConfig config){
		
		super(signature, config);
		
		mType = Constants.TYPE_DENSITY_CLUSTER;
		
		if (config.containsParam(Constants.MAX_CLUSTER_DISTANCE)) {
			mMaxDistance = (Double) config.getParam(Constants.MAX_CLUSTER_DISTANCE);
		} else {
			mMaxDistance = Constants.DEFAULT_MAX_CLUSTER_DISTANCE;
		}
		if (config.containsParam(Constants.MIN_INCLUSION_PERCENT)) {
			mMinInclusionPct = (Double) config.getParam(Constants.MIN_INCLUSION_PERCENT);
		} else {
			mMinInclusionPct = Constants.DEFAULT_MIN_INCLUSION_PERCENT;
		}
		
		FeatureNominal classFeature = (FeatureNominal)signature.getClassFeature();
		ArrayList<String> classValues = classFeature.getValues();
		
		mNumTrains = new HashMap<String, Integer>();
		mCentroids = new HashMap<String, double[]>();
		for (String classValue : classValues) {
			mCentroids.put(classValue, new double[mSignature.size() - 1]);
			mNumTrains.put(classValue, 0);
		}

	}

	@Override
	public void train(ArrayList<Instance> instances) throws MLException {

		if (Constants.DEBUG) Log.d(TAG, "train with "+instances.size()+" instances");
		
		// Remove outliers (density based)
		String curLabel;
		double curCoordValues[] = new double[mSignature.size()-1];
		
		for (Iterator<Instance> curIter = instances.iterator(); curIter.hasNext();) {
		
			Instance curInstance = curIter.next();
			
			if (!mSignature.checkCompliance(curInstance, true)){
				throw new MLException(MLException.INCOMPATIBLE_INSTANCE, 
						"Instance is not compatible with the dataset used for classifier construction.");					
			}
			
			curLabel = (String) curInstance.getValueAtIndex(mSignature.getClassIndex()).getValue();
			
			for(int i=0; i< mSignature.size()-1; i++) {
				curCoordValues[i] = (Double) curInstance.getValueAtIndex(i).getValue();
			}
			
			String otherLabel;
			double otherCoordValues[] = new double[mSignature.size()-1];
			
			int total = 0;
    		int totalInside = 0;
    		
			for (Iterator<Instance> otherIter = instances.iterator(); otherIter.hasNext();) {
				
				Instance otherInstance = otherIter.next();
				
				if (otherInstance != curInstance) {
					
					if (!mSignature.checkCompliance(otherInstance, true)){
						throw new MLException(MLException.INCOMPATIBLE_INSTANCE, 
								"Instance is not compatible with the dataset used for classifier construction.");					
					}
					
					otherLabel = (String) otherInstance.getValueAtIndex(mSignature.getClassIndex()).getValue();
					
					if (otherLabel.equals(curLabel)) {
					
						for(int i=0; i< mSignature.size()-1; i++) {
							otherCoordValues[i] = (Double) otherInstance.getValueAtIndex(i).getValue();
						}
						
						double distance = distance(curCoordValues, otherCoordValues);
						
						total++;
						
						if (distance< mMaxDistance) totalInside++;
					}
				}
			}
			
			if (Constants.DEBUG) Log.d(TAG, "Points: "+totalInside+"/"+total+" vs "
                    +mMinInclusionPct+"/100");
			if (total > 0) {
				if (totalInside/(double)total < (mMinInclusionPct /100.0)){
                    if (Constants.DEBUG) Log.d(TAG, "Remove instance");
					curIter.remove();					
	    		}
			}
			
		}
		// At this point only those instances that are tightly packed are in d_instanceQ
        if (Constants.DEBUG) Log.d(TAG, "Outliers removed. "+instances.size()+" instances left.");
		
		// Find cluster centroids
		double centroidCoords[];
		for (Instance curInstance : instances) {
			
			curLabel = (String) curInstance.getValueAtIndex(mSignature.getClassIndex()).getValue();
			centroidCoords = mCentroids.get(curLabel);

            if (Constants.DEBUG) Log.d(TAG, "Current instance label "+curLabel);
			
			for(int i=0; i< mSignature.size()-1; i++) {
                if (Constants.DEBUG) Log.d(TAG, "added coord "+i+ " with value "
                        +(Double) curInstance.getValueAtIndex(i).getValue());
				centroidCoords[i] += (Double) curInstance.getValueAtIndex(i).getValue();
			}
			
			mNumTrains.put(curLabel, mNumTrains.get(curLabel)+1);

		}
		
		int numTrains;
		for (String classValue : mCentroids.keySet()) {
			centroidCoords = mCentroids.get(classValue);
			numTrains = mNumTrains.get(classValue);

            if (Constants.DEBUG) Log.d(TAG, "Centroid with label "+classValue
                    +" contains " +numTrains+ " points.");
			
			for (int i=0; i< mSignature.size()-1; i++) {
				if (numTrains > 0)
					centroidCoords[i] =  centroidCoords[i]/numTrains;
				// otherwise keep them to zero
			}
			
			mCentroids.put(classValue, centroidCoords);
		}
		
	}

	@Override
	public Value classify(Instance instance) throws MLException {

		if (!mSignature.checkCompliance(instance, false)){
			throw new MLException(MLException.INCOMPATIBLE_INSTANCE, 
					"Instance is not compatible with the dataset used for classifier construction.");					
		}
		
		// Calculate the centroid that is the closest 
		double minDistance = Double.MAX_VALUE;
		
		// if not yet trained, we return the first label
		String minStringValue = ((FeatureNominal)mSignature.getClassFeature()).getValues().get(0);
		double centroidCoords[];
		double curCoordValues[] = new double[mSignature.size()-1];
		for(int i=0; i< mSignature.size()-1; i++) {
			curCoordValues[i] = (Double) instance.getValueAtIndex(i).getValue();
		}
		
		for (String classValue : mCentroids.keySet()) {
			centroidCoords = mCentroids.get(classValue);
			double curDistance = Double.MAX_VALUE;
			try {
				curDistance = distance(curCoordValues, centroidCoords);
			} catch (MLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} 
			if (curDistance < minDistance){
				minStringValue = classValue;
				minDistance = curDistance;
			}
		}
		
		return new Value(minStringValue, Value.NOMINAL_VALUE);
	}

    public HashMap<String,double[]> getCentroids(){
        return mCentroids;
    }

	@Override
	public void printClassifierInfo() {
		StringBuilder builder = new StringBuilder();
		builder.append("Classifer type: "+ mType +"\n");
		builder.append("Signature: "+ mSignature.toString()+"\n");
		builder.append("Centroids:\n");
		for(String classValue : mCentroids.keySet()) {
			double centroidCoords[] = mCentroids.get(classValue);
			String coords = "[";
			for (int i=0; i<centroidCoords.length-1;i++) {
				coords += (centroidCoords[i]+",");
			}
			coords += (centroidCoords[centroidCoords.length-1]+"]");
			builder.append(classValue+"("+ mNumTrains.get(classValue)+")\t"+coords+"\n");
		}		
		Log.i(TAG, builder.toString());
	}
}
