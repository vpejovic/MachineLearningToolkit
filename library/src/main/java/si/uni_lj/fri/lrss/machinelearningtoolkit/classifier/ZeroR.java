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
 * 
 * ZeroR classifier is not taking any features into account during the classification.
 * It merely outputs the mean value/most frequent class. 
 * Besides classification, we can use ZeroR for regression.
 * 
 * @author Veljko Pejovic, University of Birmingham, UK <v.pejovic@cs.bham.ac.uk>
 *
 */
public class ZeroR extends Classifier implements OnlineClassifier {

	private static final String TAG = "ZeroR";

	private double[] mClassCounts;
    
	private static final Object mLock = new Object();
	
	public ZeroR(Signature a_signature, ClassifierConfig a_config) {
		super(a_signature, a_config);
		Feature classFeature = mSignature.getClassFeature();
		if (classFeature.getFeatureType() == Feature.NOMINAL)
			mClassCounts = new double[((FeatureNominal)classFeature).numberOfCategories()];
		else if (classFeature.getFeatureType() == Feature.NUMERIC) 
			mClassCounts = new double[2];
		Arrays.fill(mClassCounts, 0.0);
	}

	@Override
	public void update(Instance instance) throws MLException {
		
		if (!mSignature.checkCompliance(instance, true)){
			throw new MLException(MLException.INCOMPATIBLE_INSTANCE, 
					"Instance is not compatible with the dataset used for classifier construction.");					
		}
		
		Feature classFeature = mSignature.getClassFeature();
		
		Value classValue = instance.getValueAtIndex(mSignature.getClassIndex());
		
		if (classFeature.getFeatureType() == Feature.NOMINAL) {
			int classValueInt = ((FeatureNominal)classFeature).indexOfCategory((String) classValue.getValue());
			mClassCounts[classValueInt] += 1;
		} else if (classFeature.getFeatureType() == Feature.NUMERIC) {
			mClassCounts[0] += (Double) classValue.getValue();
			mClassCounts[1] += 1;
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

	@Override
	public Value classify(Instance instance) throws MLException {
		
		if (!mSignature.checkCompliance(instance, false)){
			throw new MLException(MLException.INCOMPATIBLE_INSTANCE, 
					"Instance is not compatible with the dataset used for classifier construction.");					
		}
		if (((FeatureNominal)mSignature.getClassFeature()).getFeatureType() == Feature.NOMINAL) {
			double maxCount = 0;
			int maxValueIndex = 0;
			
			for (int i=0; i< mClassCounts.length; i++) {
				if (Constants.DEBUG) Log.d(TAG, "Class value index "+i+" count "+ mClassCounts[i]);
				if (mClassCounts[i] > maxCount) {
					maxValueIndex = i;
					maxCount = mClassCounts[i];
				}
			}
			
			return new Value(((FeatureNominal)mSignature.getClassFeature()).categoryOfIndex(maxValueIndex),
                    Value.NOMINAL_VALUE);
		} else { //it's NUMERIC
			double mean = mClassCounts[0]/ mClassCounts[1];
			return new Value(mean, Value.NUMERIC_VALUE);
		}
	}

	@Override
	public void printClassifierInfo() {
		// TODO Auto-generated method stub		
	}
}
