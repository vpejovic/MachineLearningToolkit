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

import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Instance;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.MLException;

/**
 * A classifier implements this interface if it allows iterative training.
 * 
 * @author Veljko Pejovic, University of Birmingham, UK <v.pejovic@cs.bham.ac.uk>
 *
 */
public interface OnlineClassifier {

    /**
     * Update an online classifier with the given training instance.
     * @param instance A labelled instance with which we train the classifier.
     * @throws MLException
     */
	public void update(Instance instance) throws MLException;

}
