/*******************************************************************************
 * Copyright (c) 2013, University of Birmingham, UK
 * Veljko Pejovic,  <v.pejovic@cs.bham.ac.uk>
 * 
 * 
 * This library was developed as part of the EPSRC Ubhave (Ubiquitous and Social
 * Computing for Positive Behaviour Change) Project. For more information, please visit
 * http://www.ubhave.org
 * 
 * Permission to use, copy, modify, and/or distribute this software for any purpose with
 * or without fee is hereby granted, provided that the above copyright notice and this
 * permission notice appear in all copies.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 ******************************************************************************/
package si.uni_lj.fri.lrss.machinelearningtoolkit.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Features can be nominal or numeric. 
 * 
 * @author Veljko Pejovic, University of Birmingham, UK <v.pejovic@cs.bham.ac.uk>
 *
 */
public abstract class Feature {
	
	private static final String TAG = "Feature";
	public static final int NOMINAL = 0;
	public static final int NUMERIC = 1;
	
	protected int mType;
    protected String mName;


    /**
     * Create a feature object with the given name and type.
     * @param fname Feature name.
     * @param ftype Feature type. Has to be {@link Feature#NUMERIC}.
     * @throws MLException
     */
	protected Feature(String fname, int ftype) {
		mType = ftype;
		mName = fname;
	}


	/*public Feature(String fname, int ftype, String[] fvalues) throws MLException{
        if (ftype == NUMERIC){
			throw new MLException(MLException.INCOMPATIBLE_FEATURE_TYPE,
					"Numeric features do not need a list of possible categories;" +
					" use Feature(String, int) instead."); 
		}
		mType = ftype;
		mName = fname;
		mCategories = new ArrayList<String>(Arrays.asList(fvalues));
		mCategoryIndex = new HashMap<String, Integer>();
 		for(int i=0;i<fvalues.length;i++) {
 			mCategoryIndex.put(fvalues[i], Integer.valueOf(i));
 		}
	}*/

	
	public int getFeatureType(){
		return mType;
	}

	public String name(){
		return mName;
	}

}
