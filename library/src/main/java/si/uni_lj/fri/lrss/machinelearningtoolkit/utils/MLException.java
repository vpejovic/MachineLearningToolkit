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

public class MLException extends Exception {

	public static final int INCOMPATIBLE_FEATURE_TYPE = 100;
	public static final int INCOMPATIBLE_INSTANCE = 101;
	public static final int INVALID_PARAMETER = 102;
	public static final int INVALID_STATE = 103;
	public static final int IO_ERROR = 104;
	public static final int IO_FILE_NOT_FOUND_ERROR = 105;
	public static final int CLASSIFIER_EXISTS = 200;
	
	private int mErrorCode;
	private String mMessage;
	
	public MLException(int errorCode, String message) {
		super(message);
		mMessage = message;
        mErrorCode = errorCode;
	}
	
	public int getErrorCode()
	{
		return mErrorCode;
	}

	public String getMessage()
	{
		return mMessage;
	}

}
