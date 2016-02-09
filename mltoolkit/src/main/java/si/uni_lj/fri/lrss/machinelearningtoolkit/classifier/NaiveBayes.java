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
import java.util.Arrays;
import java.util.HashMap;

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
 * Naive Bayesian classifier that supports both nominal and numeric attributes.
 * Numeric features are modelled with a Gaussian distribution. 
 * Laplace smoothing is supported for nominal attributes, so that classes with
 * high preference for a single value do not overfit. 
 * The classifier is an online classifier, i.e. training can happen iteratively. 
 * 
 * @author Veljko Pejovic, University of Birmingham, UK <v.pejovic@cs.bham.ac.uk>
 *
 */
public class NaiveBayes extends Classifier implements OnlineClassifier {

	private static final String TAG = "NaiveBayes";
	
	private final static Object mLock = new Object();
	
	// For each feature we hold the count of occurrences of every class variable value.
	// These are further bisected to the feature values in case of NOMINAL features.
	// For NUMERIC features we keep stats necessary for Gaussian distribution calculation.
	private HashMap<String, HashMap<String, double[]>> mValueCounts;

    // Holds the number of occurrences of each value that the class variable may take.
	private double[] mClassCounts;
	
    // Fixes the problem of too few occurrences in certain bins.
	private boolean mLaplaceSmoothing;


    /**
     * Creates a new Naive Bayesian classifier with the given signature and configuration.
     * @param signature Classifier signature.
     * @param config Optional configuration parameters.
     */
	public NaiveBayes(Signature signature, ClassifierConfig config) throws  MLException {
		super(signature, config);
		mType = Constants.TYPE_NAIVE_BAYES;
		
		if (config.containsParam(Constants.LAPLACE_SMOOTHING)) {
			mLaplaceSmoothing = (Boolean) config.getParam(Constants.LAPLACE_SMOOTHING);
		} else {
			mLaplaceSmoothing = Constants.DEFAULT_LAPLACE_SMOOTHING;
		}
		
		initialize();
	}

	public void initialize() throws MLException{
		
		mValueCounts = new HashMap<String, HashMap<String, double[]>>();
		ArrayList<Feature> features = mSignature.getFeatures();
		FeatureNominal classFeature = (FeatureNominal) mSignature.getClassFeature();
		ArrayList<String> classValues = classFeature.getValues();

		mClassCounts = new double[classFeature.numberOfCategories()];
		Arrays.fill(mClassCounts, 0.0);

		for(Feature feature : features){			
			HashMap<String, double[]> featureCount = new HashMap<String, double[]>();
		
			for (String classValue : classValues){										
				double[] classFeatureCounts;
				if (feature.getFeatureType() == Feature.NOMINAL) { 
					classFeatureCounts = new double[((FeatureNominal)feature).numberOfCategories()];
				}
				else if (feature.getFeatureType() == Feature.NUMERIC) {
					// For NUMERIC values we have to keep:
					// - count
					// - sum of values
					// - sum of square values 
					// so that we can get the normal distribution in the end
					classFeatureCounts = new double[3];
				}
				else {
                    throw new MLException(MLException.INCOMPATIBLE_FEATURE_TYPE,
                            "Feature type neither nominal nor numeric");
                }

				Arrays.fill(classFeatureCounts, 0.0);
				featureCount.put(classValue, classFeatureCounts);
				//Log.d(TAG, "Feature counts put for class value "+classValue);
			}
			String output = "Feature counts put for "+feature.name()+": ";
			for (String key : featureCount.keySet()) {
					String miniOutput = "";
					double classFeatureCountsTest[] = featureCount.get(key);
					for (int i=0; i<classFeatureCountsTest.length; i++) {
						miniOutput += classFeatureCountsTest[i]+",";
					}
				
					output +="{ "+key+", ["+miniOutput+"]},";					
			}
			if (Constants.DEBUG) Log.d(TAG, output);
			mValueCounts.put(feature.name(), featureCount);
		}
	}


	public void update(Instance instance) throws MLException {
				
		if (!mSignature.checkCompliance(instance, true)){
			throw new MLException(MLException.INCOMPATIBLE_INSTANCE, 
					"Instance is not compatible with the dataset used for classifier construction.");					
		}

		FeatureNominal classFeature = (FeatureNominal)mSignature.getClassFeature();
		
		Value classValue = instance.getValueAtIndex(mSignature.getClassIndex());

		if (classValue.getValueType() == Value.NUMERIC_VALUE ||
				classValue.getValueType() == Value.MISSING_VALUE)
            throw new MLException(MLException.INCOMPATIBLE_FEATURE_TYPE,
                    "Class variable has to be of type NOMINAL.");

		int classValueInt = classFeature.indexOfCategory((String) classValue.getValue());
		
		mClassCounts[classValueInt] += 1;
		
		for (int i=0; i< instance.size(); i++){
			
			double[] classFeatureCounts = (mValueCounts.get(mSignature.getFeatureAtIndex(i).name()))
					.get(classFeature.categoryOfIndex(classValueInt));
			Value featureValue = instance.getValueAtIndex(i);
			
			if (featureValue.getValueType() == Value.NOMINAL_VALUE){
                FeatureNominal currentFeature = (FeatureNominal)mSignature.getFeatureAtIndex(i);
				int featureValueCat = currentFeature.indexOfCategory((String) featureValue.getValue());				
				classFeatureCounts[featureValueCat] += 1;
				String output = "Update:"+ mSignature.getFeatureAtIndex(i).name()
						+"["+(String) classValue.getValue()+"] = {";
				for (int j=0; j<classFeatureCounts.length; j++) {
					output += classFeatureCounts[j]+",";
				}
                if (Constants.DEBUG) Log.d(TAG, output + "}");
			}
			if (featureValue.getValueType() == Value.NUMERIC_VALUE){
				classFeatureCounts[0] += 1; // count				
				classFeatureCounts[1] += (Double)featureValue.getValue(); // value sum
				classFeatureCounts[2] += Math.pow((Double)featureValue.getValue(),2); // value square sum
                if (Constants.DEBUG) Log.d(TAG, "Update:"+ mSignature.getFeatureAtIndex(i).name()
						+"["+(String) classValue.getValue()+"] = "
						+"{"+classFeatureCounts[0]+","+classFeatureCounts[1]+","+classFeatureCounts[2]+"}");
			}
			// Do nothing for a missing value.
		}
	}

	
	@Override
	public void train(ArrayList<Instance> a_instances) throws MLException {
		
		for(Instance a_instance : a_instances){
			synchronized (mLock) {
				this.update(a_instance);
			}
		}
	}
	
	
	public double[] getDistribution(Instance instance) throws MLException {
		
		if (!mSignature.checkCompliance(instance, false)){
			throw new MLException(MLException.INCOMPATIBLE_INSTANCE, 
					"Instance is not compatible with the dataset used for classifier construction.");					
		}
		
		FeatureNominal classFeature = (FeatureNominal) mSignature.getClassFeature();
		ArrayList<String> classValues = classFeature.getValues();
		double[] classPriors = new double[classValues.size()];
		double[] classPosteriors = new double[classValues.size()];
		
		int classCountsTotal = 0;
		for (int j=0; j< mClassCounts.length; j++) classCountsTotal += mClassCounts[j];
		for (int j=0; j<classPriors.length; j++) {
			if (classCountsTotal == 0) {
				classPriors[j] = 1.0/classPriors.length; 
			} else {
				classPriors[j] = mClassCounts[j]/classCountsTotal;
			}
			classPosteriors[j] = classPriors[j];
			
		}

        if (Constants.DEBUG) Log.d(TAG, "Class values: "+classValues);
		String outputPrior="[";
		for (int i=0; i<classPriors.length;i++) {
			outputPrior += classPriors[i]+",";
		}
		if (Constants.DEBUG) Log.d(TAG, "Class priors: "+outputPrior);
		
		for (int i=0; i<instance.size(); i++){
			Value featureValue = instance.getValueAtIndex(i);
			Feature feature = mSignature.getFeatureAtIndex(i);
			// for every feature (a specific value of it) we get a prob of each class
			double[] classFeatureProbs = new double[classValues.size()];
			
			// A map of all class values -> feature counts
			HashMap<String, double[]> featureCounts = mValueCounts.get(feature.name());

		
			for (String classValue : classValues){
				
				double[] classFeatureCounts = featureCounts.get(classValue);		
				int indexOfClassValue = classFeature.indexOfCategory(classValue);
				double classFeatureTotal = 0;
				
				if (featureValue.getValueType() == Value.NOMINAL_VALUE) {
                    FeatureNominal featureNom = (FeatureNominal) feature;
					String printFeatureCounts = "Class feature counts: ";
                    for (double classFeatureCount : classFeatureCounts) {
                        classFeatureTotal += classFeatureCount;
                        printFeatureCounts += classFeatureCount + ", ";
                    }
					if (Constants.DEBUG) Log.d(TAG, printFeatureCounts);

					int featureValueIndex = featureNom.indexOfCategory((String) featureValue.getValue());
                    if (Constants.DEBUG) Log.d(TAG, "Feature value "+featureValue.getValue().toString()+" index: "+featureValueIndex);

					if (mLaplaceSmoothing){
						classFeatureProbs[indexOfClassValue]=classFeatureCounts[featureValueIndex]+1/
								(classFeatureTotal + featureNom.numberOfCategories());
					}
					else {
						if (classFeatureTotal > 0){
							classFeatureProbs[indexOfClassValue]=classFeatureCounts[featureValueIndex]/classFeatureTotal;
						}
					}
                    if (Constants.DEBUG) Log.d(TAG, "classFeatureProbs["+indexOfClassValue+"]= "+classFeatureProbs[indexOfClassValue]);
					classPosteriors[indexOfClassValue] *= classFeatureProbs[indexOfClassValue];
					
				} else if (featureValue.getValueType() == Value.NUMERIC_VALUE) {
					
					double mean;
					double stdDev;
					// ignore features for which we have no data (those will have normalProb = 1) 
					double normalProbability = 1; 
					double featureValueDouble = (Double)featureValue.getValue();
					
					if (classFeatureCounts[0] > 0) {
						mean = classFeatureCounts[1]/classFeatureCounts[0];
						stdDev = Math.sqrt(classFeatureCounts[2] - Math.pow(mean,2));
						normalProbability = Math.exp(Math.pow(featureValueDouble - mean ,2))/(2*stdDev*Math.sqrt(2*Math.PI));
						// NOTE: if the current value equals the mean the normal probability goes to infinity;
						// to prevent this we, cap it to 1.0.
						if (Double.isInfinite(normalProbability)) normalProbability = 1.0;
						
					}

                    if (Constants.DEBUG) Log.d(TAG, "calc for: "+ classValue
							+" and " + feature.name()
							+" resulting probability "+normalProbability+" total "+classPosteriors[indexOfClassValue]);

					classPosteriors[indexOfClassValue] *= normalProbability;
				}
			}
		}
		
		String outputPosterior="[";
		for (int i=0; i<classPosteriors.length;i++) {
			outputPosterior += classPosteriors[i]+",";
		}
		outputPosterior += "]";
		if (Constants.DEBUG) Log.d(TAG, "Class posteriors: "+outputPosterior);
		return classPosteriors;
	}


	@Override
	public Value classify(Instance a_instance) throws MLException {
		synchronized (mLock)
		{
			double[] classDistribution = getDistribution(a_instance);
			double maxAposteriori = 0;
			int maxAposterioriIndex = -1;
			for (int i=0; i<classDistribution.length; i++){
				if (classDistribution[i] > maxAposteriori){
					maxAposteriori = classDistribution[i];
					maxAposterioriIndex = i;
				}
			}
			
			// When the classifier is not yet trained we return the first class value
			if (maxAposterioriIndex == -1) {
				maxAposterioriIndex = 0;
			}

            if (mSignature.getClassFeature().getFeatureType() == Feature.NOMINAL) {
                return new Value(((FeatureNominal) mSignature.getClassFeature())
                        .categoryOfIndex(maxAposterioriIndex),Value.NOMINAL_VALUE);
            }
            else {
                throw new MLException(MLException.INCOMPATIBLE_FEATURE_TYPE, "class feature must be nominal");
            }
		}
	}

	@Override
	public void printClassifierInfo() {

        FeatureNominal classFeature = (FeatureNominal)mSignature.getClassFeature();
		StringBuilder builder = new StringBuilder();
		builder.append("Classifier type: "+ mType +"\n");
		builder.append("Signature: "+ mSignature.toString()+"\n");
		builder.append("Class feature value counts: ");
		for (int i=0; i< classFeature.getValues().size(); i++){
			
			builder.append("["+ classFeature.getValues().get(i)+":"+ mClassCounts[i]+"]");
		}
		builder.append("\nOther feature value counts: \n");
		for (String featureName : mValueCounts.keySet()){
			builder.append(featureName+" ");
			HashMap<String,double[]> counts = mValueCounts.get(featureName);
			for (String classFeatureValue : counts.keySet()) {
				builder.append("["+classFeatureValue+":");
				double[] values = counts.get(classFeatureValue);
				for (int i=0; i<values.length; i++) {
					builder.append(values[i]+",");
				}
				builder.append("],");
			}
			builder.append("\n");
		}
        Log.i(TAG, builder.toString());
	}
}
