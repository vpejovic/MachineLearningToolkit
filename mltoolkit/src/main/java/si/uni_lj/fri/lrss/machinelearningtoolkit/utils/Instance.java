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

/**
 * Instance is merely a list of Values. It should be used, and comply with, 
 * the signature of the classifier.
 * 
 * @author Veljko Pejovic, University of Birmingham, UK <v.pejovic@cs.bham.ac.uk>
 *
 */
public class Instance {

	private ArrayList<Value> mValues;
	
	public Instance(int a_numFeatures){
		mValues = new ArrayList<Value>(a_numFeatures);
	}
	
	public Instance(){
		this(0);
	}
	
	public Instance(ArrayList<Value> featureValues){
		mValues = featureValues;
	}
	
	public void addValue(Value value){
		mValues.add(value);
	}
	
	public Value getValueAtIndex(int i){
		return mValues.get(i);
	}
	
	public void setValueAtIndex(int i, Value value) throws IndexOutOfBoundsException {
		mValues.set(i, value);
	}
	
	
	public int size(){
		return mValues.size();
	}
}
