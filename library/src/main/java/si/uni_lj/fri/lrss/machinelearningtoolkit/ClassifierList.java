package si.uni_lj.fri.lrss.machinelearningtoolkit;


import android.util.Log;

import java.util.HashMap;

import si.uni_lj.fri.lrss.machinelearningtoolkit.classifier.Classifier;
import si.uni_lj.fri.lrss.machinelearningtoolkit.classifier.DensityClustering;
import si.uni_lj.fri.lrss.machinelearningtoolkit.classifier.ID3;
import si.uni_lj.fri.lrss.machinelearningtoolkit.classifier.NaiveBayes;
import si.uni_lj.fri.lrss.machinelearningtoolkit.classifier.ZeroR;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.ClassifierConfig;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Constants;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.MLException;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Signature;

/**
 *
 * Takes care of classifier instantiation and registration.
 * Every classifier that is created has a unique name.
 *
 * @author Veljko Pejovic, University of Birmingham, UK <v.pejovic@cs.bham.ac.uk>
 *
 */
public class ClassifierList {

    private static final String TAG = "ClassifierList";

    private HashMap<String, Classifier> mNamedClassifiers;

    //private final Random d_keyGenerator;

    protected ClassifierList(){
        if (Constants.DEBUG) Log.d(TAG, "ClassifierList empty constructor");
        mNamedClassifiers = new HashMap<String, Classifier>();
        //d_keyGenerator = new Random();
    }

    private static Classifier createClassifier (
            int type,
            Signature signature,
            ClassifierConfig config) throws MLException{

        if (Constants.DEBUG) Log.d(TAG, "createClassifier");

        switch (type) {
            case Constants.TYPE_NAIVE_BAYES:
                if (Constants.DEBUG) Log.d(TAG, "create NaiveBayes");
                return new NaiveBayes(signature, config);
            case Constants.TYPE_ID3:
                if (Constants.DEBUG) Log.d(TAG, "create ID3");
                return new ID3(signature, config);
            case Constants.TYPE_DENSITY_CLUSTER:
                if (Constants.DEBUG) Log.d(TAG, "create DensityClustering");
                return new DensityClustering(signature, config);
            case Constants.TYPE_ZERO_R:
                if (Constants.DEBUG) Log.d(TAG, "create ZeroR");
                return new ZeroR(signature, config);
            default:
                if (Constants.DEBUG) Log.d(TAG, "create default (NaiveBayes)");
                return new NaiveBayes(signature, config);
        }
    }

    protected void removeClassifier(String a_classifierID) {
        if (mNamedClassifiers.containsKey(a_classifierID)) {
            mNamedClassifiers.remove(a_classifierID);
        }
    }

    protected Classifier getClassifier(String a_classifierID)
    {
        if (mNamedClassifiers.containsKey(a_classifierID)) {
            return mNamedClassifiers.get(a_classifierID);
        } else {
            return null;
        }
    }

    protected synchronized Classifier addClassifier(
            int type, Signature signature, ClassifierConfig config, String name) throws MLException {

        if (Constants.DEBUG) Log.d(TAG, "addClassifier");

        Classifier classifier = createClassifier(type, signature, config);
        mNamedClassifiers.put(name, classifier);
        return classifier;
    }

}
