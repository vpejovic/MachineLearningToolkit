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
package si.uni_lj.fri.lrss.machinelearningtoolkit.utils;

import java.util.ArrayList;

import android.util.Log;

/**
 * Signature defines features used by a classifier, their names and the 
 * index of the class feature. Instances of data (which consist of values) 
 * need to correspond to the signature of the classifier they are used for.
 * Signature exposes a method for checking that compliance. 
 * 
 * @author Veljko Pejovic, University of Birmingham, UK <v.pejovic@cs.bham.ac.uk>
 *
 */
public class Signature {

	private static final String TAG = "Signature";
	
	private ArrayList<Feature> mFeatures;
	   
    private int mClassIndex;

	public Signature(ArrayList<Feature> features, int classIndex){
		mFeatures = features;
		mClassIndex = classIndex;

	}
	
	public Signature(ArrayList<Feature> features){
		this(features, features.size() - 1);
	}
	
	public void setClassIndex(int classIndex){
		mClassIndex = classIndex;
	}
	
	public int getClassIndex(){
		return mClassIndex;
	}
	
	public Feature getClassFeature(){
		return getFeatureAtIndex(getClassIndex());
	}
	
	public Feature getFeatureAtIndex(int i){
		return mFeatures.get(i);
	}
	
	public ArrayList<Feature> getFeatures(){
		return mFeatures;
	}
	
	public int size() {
		return mFeatures.size();
	}

	public boolean checkCompliance(Instance instance, boolean training) {
		Log.d(TAG, "checkInstanceCompliance");
		int checkSize = instance.size();
		
		// Instances that are used for training should have the exact same features as the signature.
		// Those that are about to be classified, should have one feature less -- the class feature.
		if (!training) {
			checkSize++;
		}
		
		if (checkSize != this.getFeatures().size()){
			Log.d(TAG, "Expected number of features: "+this.getFeatures().size()+" got "+checkSize);
			for (int i=0; i<instance.size(); i++){
				Log.d(TAG, "instance value "+instance.getValueAtIndex(i).getValue());
			}
			return false;
		}
		
		for (int i=0; i<instance.size(); i++){
			Log.d(TAG, "instance type: "+instance.getValueAtIndex(i).getValueType()+" feature type: "+this.getFeatureAtIndex(i).getFeatureType());
			
			if (this.getFeatureAtIndex(i).getFeatureType() != instance.getValueAtIndex(i).getValueType()
					&& (instance.getValueAtIndex(i).getValueType() != Value.MISSING_VALUE)){
				return false;
			}
		}
		return true;	
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		for (Feature f : mFeatures){
			builder.append("["+f.name()+"("+f.getFeatureType()+")"+"], ");
		}
		return builder.toString();
	}
}
