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

import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.ClassifierConfig;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Instance;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.MLException;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Signature;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Value;

/**
 * Every classifier has a signature that defines the features it uses,
 * as well as the class feature. It needs to support training and classification. 
 * 
 * @author Veljko Pejovic, University of Birmingham, UK <v.pejovic@cs.bham.ac.uk>
 *
 */
public abstract class Classifier {
	
	protected Signature mSignature;
	
	protected int mType;
	
	protected ClassifierConfig mConfig;

    protected boolean mTrained;

	private static final String TAG = "Classifier";

	public Classifier(Signature a_signature, ClassifierConfig a_config) {
		mSignature = a_signature;
		mConfig = a_config;
	}

	/**
	 * Train the classifier with labelled data instances.
	 * @param instances Labelled data instances. The instances have to correspond to the classifier
	 *                  signature.
	 * @throws MLException
	 */
	public abstract void train(ArrayList<Instance> instances) throws MLException;

	/**
	 * Classify an unlabelled instance.
	 * @param instance Instance to be classified.
	 * @return The inferred label corresponding to the given instance.
	 * @throws MLException
	 */
	public abstract Value classify(Instance instance) throws MLException;

	public abstract void printClassifierInfo();

    public boolean isTrained() {
        return mTrained;
    }
	
}
