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
import java.util.Iterator;
import java.util.Map.Entry;

import android.util.Log;

import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.ClassifierConfig;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Constants;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Feature;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.FeatureNominal;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Instance;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.MLException;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Signature;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Value;

// TODO: Nominal attributes are supported for now.
// TODO: How to handle missing attributes?
// TODO: Pruning?

/**
 * ID3, a tree-based classifier according to: 
 * Quinlan, J. R. 1986. Induction of Decision Trees. Mach. Learn. 1, 1 (Mar. 1986), 81-106
 * This implementation supports nominal attributes only.
 * 
 * @author Veljko Pejovic, University of Birmingham, UK <v.pejovic@cs.bham.ac.uk>
 *
 */
public class ID3 extends Classifier{

	/*
	 * Algorithm pseudocode (from Wikipedia):
	ID3 (Examples, Target_Attribute, Attributes)
    Create a root node for the tree
    If all examples are positive, Return the single-node tree Root, with label = +.
    If all examples are negative, Return the single-node tree Root, with label = -.
    If number of predicting attributes is empty, then Return the single node tree Root,
    with label = most common value of the target attribute in the examples.
    Otherwise Begin
        A <- The Attribute that best classifies examples.
        Decision Tree attribute for Root = A.
        For each possible value, v_i, of A,
            Add a new tree branch below Root, corresponding to the test A = v_i.
            Let Examples(v_i) be the subset of examples that have the value v_i for A
            If Examples(v_i) is empty
                Then below this new branch add a leaf node with label = most common target value in the examples
            Else below this new branch add the subtree ID3 (Examples(v_i), Target_Attribute, Attributes – {A})
    End
    Return Root
    */
	
	private Feature mBestFeature;
	private int mBestFeatureIndex;
	private int[] mCandidateFeatures;
	private Value mMajorValue;
	private boolean mIsLeaf;
	private HashMap<Object, ID3> mSubtrees;

	private static final String TAG = "ID3";

	private static int[] allFeaturesArray(Signature signature) {
		int[] candidateFeatures = new int[signature.getFeatures().size()];
		Arrays.fill(candidateFeatures, 1);
		candidateFeatures[signature.getClassIndex()] = 0;
		return candidateFeatures;
	}

	/**
	 * Creates a new ID3 classifier with the given signature and configuration.
	 * @param signature Classifier signature.
	 * @param config Optional configuration parameters.
	 */
	public ID3(Signature signature, ClassifierConfig config) {
		this(signature, config, allFeaturesArray(signature));
	}
	
	private ID3(Signature signature, ClassifierConfig config, int[] candidateFeatures) {
		super(signature, config);
		mType = Constants.TYPE_ID3;
		mSubtrees = new HashMap<Object, ID3>();
		mCandidateFeatures = candidateFeatures;
		mIsLeaf = true;
		mMajorValue = null;
		mBestFeature = null;
		mBestFeatureIndex = 0;
	}

	@Override
	public void train(ArrayList<Instance> instances) throws MLException {
		// Calculate stats such as:
		FeatureNominal classFeature = (FeatureNominal) mSignature.getClassFeature();
		int[] classCounts = new int[classFeature.numberOfCategories()];	
		Arrays.fill(classCounts, 0);
		
		// if all Instances belong to a single class
		for (Instance i : instances) {
			Value classValue = i.getValueAtIndex(mSignature.getClassIndex());
			int classValueInt = classFeature.indexOfCategory((String) classValue.getValue());
			classCounts[classValueInt] += 1;			
		}
		int NZcounter = 0, nonZeroClassValueIndex = 0;
		for (int i = 0; i < classCounts.length; i ++)
		    if (classCounts[i] > 0) {
		    	NZcounter ++;
		    	nonZeroClassValueIndex = i;
		    }
		if (NZcounter == 1){
			mIsLeaf = true;
			mMajorValue = new Value(classFeature
                    .categoryOfIndex(nonZeroClassValueIndex), Value.NOMINAL_VALUE);
			return;
		}
		
		// If number of predicting attributes is empty, then Return the single node tree Root,
	    // with label = most common value of the target attribute in the examples.
		int numCandidateFeatures = 0;
		for (int indicator : mCandidateFeatures)
			numCandidateFeatures += indicator;
		if (numCandidateFeatures == 0) {
			mIsLeaf = true;
			int maxClassValueInt = 0;
			int maxClassValueCount = 0;
			for (int i=0; i<classCounts.length; i++){
				if (maxClassValueCount < classCounts[i]) {
					maxClassValueInt = i;
					maxClassValueCount = classCounts[i];
				}
			}
			mMajorValue = new Value(classFeature
                    .categoryOfIndex(maxClassValueInt), Value.NOMINAL_VALUE);
		}
		
		else {
			// Calculate information gain for each attribute
			double IG[] = new double[mSignature.getFeatures().size()];
			double maxIG = -1;
			int maxIGindex = 0;
			HashMap<String, ArrayList<Instance>> maxSubsets = null;
			
			double totalSetEntropy = calculateEntropy(instances);
			// H(S) - sum(p(t)*H(t))_for_attribute_A_the_data_is_split_in_T_sets
			// H(t) = - sum(p(x)log(p(x))) where x in X (set of class values)
			//TODO: There should be a faster way to check if there are any 
			// interesting attributes left, before the entropy is calculated.

			for(int i=0; i< mSignature.getFeatures().size(); i++){
				
				Feature feature = mSignature.getFeatureAtIndex(i);
				
				if (feature.getFeatureType() == Feature.NOMINAL && mCandidateFeatures[i]==1) {
					FeatureNominal featureNom = (FeatureNominal) feature;
					HashMap<String, ArrayList<Instance>> subsets = new HashMap<String, ArrayList<Instance>>();
					int classFeatureCounts[] = new int[featureNom.numberOfCategories()];
					Arrays.fill(classFeatureCounts, 0);
					
					for (Instance instance : instances) {
						
						Value featureValue = instance.getValueAtIndex(i);
						String featureValueName = (String) featureValue.getValue();
						int featureValueInt = featureNom.indexOfCategory(featureValueName);
						
						classFeatureCounts[featureValueInt] += 1;
						
						ArrayList<Instance> subset = subsets.get(featureValueName);
						if (subset == null) {
							subset = new ArrayList<Instance>();
						}
						subset.add(instance);
						subsets.put(featureValueName, subset);
					}
					
					
					Iterator<Entry<String, ArrayList<Instance>>> it = subsets.entrySet().iterator();
					double sumEntropies = 0;
					while (it.hasNext()) {
						Entry<String, ArrayList<Instance>> pairs = it.next();
						double pFeatureValue = pairs.getValue().size()/(double)instances.size();
						double entropy = calculateEntropy(pairs.getValue());
						sumEntropies += pFeatureValue * entropy;
					}
					double IGvalue = totalSetEntropy - sumEntropies;
					
					if (IGvalue > maxIG) {
						maxIG = IGvalue;
						maxIGindex = i;
						maxSubsets = subsets;
					}
					IG[i] = IGvalue;
				}
			}
			// Pick the best attribute 
			mBestFeature = mSignature.getFeatureAtIndex(maxIGindex);
			mBestFeatureIndex = maxIGindex;
			mIsLeaf = false;
			
			// TODO: what happens if some of the deciding feature values are not observed
			// in the training set? Right now we point it to the majority class.

            if (mBestFeature.getFeatureType() == Feature.NOMINAL) {
                ArrayList<String> featureValueList = ((FeatureNominal)mBestFeature).getValues();
                int candidateFeatures[] = (int[]) mCandidateFeatures.clone();
                candidateFeatures[maxIGindex] = 0;

                for (String value : featureValueList) {
                    ID3 subTree = null;
                    // TODO: Check what happens if maxSubsets is null
                    if (maxSubsets.containsKey(value)) {
                        subTree = new ID3(mSignature, mConfig, candidateFeatures);
                        subTree.train(maxSubsets.get(value));

                    } else {
                        // Shortcut to the majority class leaf.
                        Arrays.fill(candidateFeatures, 0);
                        subTree = new ID3(mSignature, mConfig, candidateFeatures);
                    }
                    mSubtrees.put(value, subTree);
                }
            }
		}
		// if no attributes left -> return majority class

	}

	// H(t) = - sum(p(x)log(p(x))) where x in X (set of class values)
	private double calculateEntropy (ArrayList<Instance> instances) {
		
		double entropy = 0;
		
		FeatureNominal classFeature = (FeatureNominal) mSignature.getClassFeature();

		int classCounts[] = new int[classFeature.numberOfCategories()];
		int classCountsTotal = 0;
		Arrays.fill(classCounts, 0);
		
		for (Instance i : instances) {
			Value classValue = i.getValueAtIndex(mSignature.getClassIndex());
			int classValueInt = classFeature.indexOfCategory((String) classValue.getValue());
			classCounts[classValueInt] += 1;
		}

		for (int j=0; j<classCounts.length; j++) classCountsTotal += classCounts[j];
		for (int j=0; j<classCounts.length; j++) {
			double probabilityFeatureValue = (double)classCounts[j]/classCountsTotal;
			if (probabilityFeatureValue > 0) {
				entropy -= (probabilityFeatureValue)*Math.log(probabilityFeatureValue);
			}
		}
		
		return entropy;
	}
	
	@Override
	public Value classify(Instance a_instance) throws MLException {

		if (!mSignature.checkCompliance(a_instance, false)){
			throw new MLException(MLException.INCOMPATIBLE_INSTANCE, 
					"Instance is not compatible with the dataset used for classifier construction.");					
		}

		if (mIsLeaf) {
			return mMajorValue;
		} else {
			Value v = a_instance.getValueAtIndex(mBestFeatureIndex);
			ID3 nextTree = mSubtrees.get(v.getValue());
			return nextTree.classify(a_instance);
		}
	}
	
	private String print(int depth) {
		String output;
		if (mIsLeaf) {
			output = mMajorValue.getValue().toString()+"\n";
		} else {
			output = mBestFeature.name()+"\n";
			Iterator<Entry<Object, ID3>> it = mSubtrees.entrySet().iterator();
			while(it.hasNext()){
				Entry<Object, ID3> pair = it.next();
				String tmp = "\t";
				for (int i=0; i<depth; i++){
					tmp += "\t";
				}
				output += (tmp+pair.getKey().toString()+" -> "+pair.getValue().print(depth+1)+"\n");
			}
		}
		return output;		
	}
	
	@Override
	public void printClassifierInfo() {
		Log.i(TAG, this.print(0));
	}
}
