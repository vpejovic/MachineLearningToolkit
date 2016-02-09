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

/**
 * Values can be double for numeric, int for nominal, 
 * and a special NaN for missing values.
 * 
 * @author Veljko Pejovic (v.pejovic@cs.bham.ac.uk)
 *
 */
public class Value {

	public static final int NOMINAL_VALUE=0;
	public static final int NUMERIC_VALUE=1;
	public static final int MISSING_VALUE=2;

	private Object mValue;
	private int mType;

	public Value(Object value, int type){
		switch (type) {
			case NOMINAL_VALUE: mValue = value;
								mType = NOMINAL_VALUE;
								break;
			case NUMERIC_VALUE: mValue = value;
								mType = NUMERIC_VALUE;
								break;
			case MISSING_VALUE: mValue = Double.NaN;
								mType = MISSING_VALUE;
								break;			
		}
	}
	
	public int getValueType(){
		return mType;
	}
	
	public Object getValue(){
		return mValue;
	}
}
