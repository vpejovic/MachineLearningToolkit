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

public class Constants {
	
	public static final int TYPE_ZERO_R = 1000;
	public static final int TYPE_NAIVE_BAYES = 1001;
	public static final int TYPE_BAYES_NET = 1002;
	public static final int TYPE_ID3 = 1003;
	public static final int TYPE_DENSITY_CLUSTER = 1004;
	
	public static final String CLASSIFIER_STORAGE_FILE = "classifiers.json";

	// Config params
	
	// Density clustering
	public static final String MAX_CLUSTER_DISTANCE = "maxClusterDistance";
	public static final String MIN_INCLUSION_PERCENT = "minInclusionPercent";

	public static final double DEFAULT_MAX_CLUSTER_DISTANCE = 1; // in km if GPS
	public static final double DEFAULT_MIN_INCLUSION_PERCENT = 50.0; 
	
	// Naive Bayes
	public static final String LAPLACE_SMOOTHING = "laplaceSmoothing";
	
	public static final boolean DEFAULT_LAPLACE_SMOOTHING = true;
	
	public static final boolean DEBUG = false;

	
}
