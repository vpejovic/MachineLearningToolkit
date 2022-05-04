package si.uni_lj.fri.lrss.machinelearningtoolkit;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.lang.reflect.Type;
import java.util.Arrays;

import android.content.Context;
import android.os.Environment;
import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;

import si.uni_lj.fri.lrss.machinelearningtoolkit.ClassifierList;
import si.uni_lj.fri.lrss.machinelearningtoolkit.classifier.Classifier;
import si.uni_lj.fri.lrss.machinelearningtoolkit.classifier.DensityClustering;
import si.uni_lj.fri.lrss.machinelearningtoolkit.classifier.ID3;
import si.uni_lj.fri.lrss.machinelearningtoolkit.classifier.NaiveBayes;
import si.uni_lj.fri.lrss.machinelearningtoolkit.classifier.ZeroR;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Constants;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.MLException;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.Signature;
import si.uni_lj.fri.lrss.machinelearningtoolkit.utils.ClassifierConfig;

/**
 * Deals with instantiating classifiers, sending training/test data
 * to the right classifier, namely the one that an application has instantiated earlier.
 * Has methods that allow it to save classifiers to a file, and load them from a file.
 * This should be used when a service that uses the manager is (re)started/destroyed.
 * Only one MachineLearningManager exists per context to ensure consistency.
 *
 * @author Veljko Pejovic, University of Birmingham, UK <v.pejovic@cs.bham.ac.uk>
 *
 */
public class MachineLearningManager {

    private static final String TAG = "MLManager";

    private static MachineLearningManager sManager;
    private final ClassifierList mClassifiers;
    private final Context mContext;
    private static final Object sLock = new Object();

    /**
     * Instantiates a new manager to be used for communication between the ML library and the
     * overlying application. If the manager for the given context has been instantiated already,
     * the existing objects will be returned. The returned manager is a singleton class for the
     * given context.
     * @param context Context for which the manager is to be instantiated.
     * @return Manager for the given context.
     * @throws MLException
     */
    public static MachineLearningManager getMLManager(Context context) throws MLException {

        if (context == null) {
            throw new MLException(MLException.INVALID_PARAMETER,
                    " Invalid parameter, context object passed is null");
        }
        if (sManager == null)
            synchronized (sLock)
            {
                if (sManager == null) sManager = new MachineLearningManager(context);
            }
        return sManager;
    }

    private MachineLearningManager(Context context) throws MLException {
        mContext = context;
        // automatic loading if classifiers exist on the device
        if (Arrays.asList(mContext.fileList()).contains(Constants.CLASSIFIER_STORAGE_FILE)){
            mClassifiers = loadFromPersistent();
        }
        else{
            mClassifiers = new ClassifierList();
        }
    }

    /**
     * Instantiates a new, or returns an existing classifier with the given name and properties.
     * Properties include classifier type (available options can be found in {@link Constants}),
     * classifier signature ({@link Signature}) and classifier configuration
     * ({@link ClassifierConfig}).
     * @param type Type of classifier. One of the options from {@link Constants}).
     * @param signature Classifier signature
     * @param config Classifier configuration, if any.
     * @param name Unique classifier name.
     * @return A new, or an existing classifier if one with the given name is already instantiated.
     */
    public Classifier addClassifier(int type, Signature signature,
                                    ClassifierConfig config, String name) throws MLException {

        Log.d(TAG, "addClassifier");

        Classifier cls = mClassifiers.getClassifier(name);

        // TODO: Expose classifier properties so that we can check
        // if the existing classifier is the same as the one we require.
        if (cls != null) {
            //Log.d(TAG, "return existing classifier");
            return cls;
        }
        //throw new MLException(MLException.CLASSIFIER_EXISTS, "Classifier "+name+" already exists.");
        //Log.d(TAG, "return brand new classifier");
        return mClassifiers.addClassifier(type, signature, config, name);
    }

    /**
     * Removes a classifier if it exists. If a classifier with the given name does not exist, nothing
     * will happen.
     * @param name Classifier name.
     */
    public void removeClassifier(String name){
        mClassifiers.removeClassifier(name);
    }

    /**
     * Returns a classifier with the given name, or null if such a classifier doesn't exist.
     * @param name Classifier name.
     * @return An instance of the classifier with the given name.
     */
    public Classifier getClassifier(String name){
        return mClassifiers.getClassifier(name);
    }

    public String getJSON() {
        Gson gson = new Gson();
        return gson.toJson(mClassifiers);
    }

    /**
     * Saves classifiers to an external json-formatted file.
     * @param filename Desired classifier file name.
     */
    public void saveToPersistentExternal(String filename) throws MLException {
        Gson gson = new Gson();
        String jsonString = gson.toJson(mClassifiers);

        try {
            String root = Environment.getExternalStorageDirectory().toString();
            File file = new File(root + filename);
            FileOutputStream fos = new FileOutputStream(file);
            OutputStreamWriter osw = new OutputStreamWriter(fos);
            osw.write(jsonString);
            osw.flush();
            osw.close();
        } catch (FileNotFoundException e) {
            throw new MLException(MLException.IO_FILE_NOT_FOUND_ERROR, "File "+filename+" not found.");
        } catch (IOException e) {
            throw new MLException(MLException.IO_ERROR, "IO exception while writing "+filename+".");
        }
    }

    /**
     * Saves classifiers to a persistent internal file. The overlying application should call this
     * method before the application is closed/destroyed.
     */
    public void saveToPersistent() throws MLException {
        Gson gson = new Gson();
        String jsonString = gson.toJson(mClassifiers);

        try {
            FileOutputStream fos = mContext
                    .openFileOutput(Constants.CLASSIFIER_STORAGE_FILE, Context.MODE_PRIVATE);
            OutputStreamWriter osw = new OutputStreamWriter(fos);
            osw.write(jsonString);
            osw.flush();
            osw.close();
        } catch (FileNotFoundException e) {
            throw new MLException(MLException.IO_FILE_NOT_FOUND_ERROR, "Classifier file not found.");
        } catch (IOException e) {
            throw new MLException(MLException.IO_ERROR, "IO exception while writing internal storage.");
        }
    }

    /**
     * Loads classifiers from a given file residing on the external storage.
     * @param filename JSON formatted file with classifier information.
     * @return List of classifiers loaded from the file.
     */
    public ClassifierList loadFromExternalPersistent(String filename) throws MLException {

        StringBuilder jsonString = new StringBuilder();

        try {
            File sdcard = Environment.getExternalStorageDirectory();
            File file = new File(sdcard,filename);
            FileReader fr = new FileReader(file);
            BufferedReader br = new BufferedReader(fr);

            String line;
            while ((line = br.readLine()) != null) {
                jsonString.append(line);
                Log.d(TAG, "Read "+line);
            }
            br.close();
        } catch (FileNotFoundException e) {
            throw new MLException(MLException.IO_FILE_NOT_FOUND_ERROR, "File "+filename+" not found.");
        } catch (IOException e) {
            throw new MLException(MLException.IO_ERROR, "IO exception while writing "+filename+".");
        }

        Gson gson = new GsonBuilder()
                .registerTypeHierarchyAdapter(Classifier.class, new ClassifierAdapter())
                .create();

        return (ClassifierList) gson.fromJson(jsonString.toString(), ClassifierList.class);
    }

    /**
     * Loads classifiers from a persistent internal file. The method is called automatically when
     * the ML manager is instantiated.
     */
    public ClassifierList loadFromPersistent() throws MLException {

        StringBuilder jsonString = new StringBuilder();

        try {
            FileInputStream is = mContext.openFileInput(Constants.CLASSIFIER_STORAGE_FILE);
            BufferedReader br = new BufferedReader(new InputStreamReader(is));

            String line;
            while ((line = br.readLine()) != null) {
                jsonString.append(line);
            }
            br.close();
        } catch (FileNotFoundException e) {
            throw new MLException(MLException.IO_FILE_NOT_FOUND_ERROR, "Classifier file not found.");
        } catch (IOException e) {
            throw new MLException(MLException.IO_ERROR, "IO exception while writing internal storage.");
        }

        Gson gson = new GsonBuilder()
                .registerTypeHierarchyAdapter(Classifier.class, new ClassifierAdapter())
                .create();

        return (ClassifierList) gson.fromJson(jsonString.toString(), ClassifierList.class);
    }

    static class ClassifierAdapter implements JsonDeserializer<Classifier> {

        Gson gson;

        ClassifierAdapter(){
            GsonBuilder gsonBuilder = new GsonBuilder();
            gson = gsonBuilder.create();
        }

        public Classifier deserialize(JsonElement elem, Type type, JsonDeserializationContext context)
                throws JsonParseException {
            Classifier result = null;

            JsonObject object = elem.getAsJsonObject();
            int intType = object.get("mType").getAsInt();
            switch(intType){
                case Constants.TYPE_NAIVE_BAYES:
                    result = gson.fromJson(elem, NaiveBayes.class);
                    break;
                case Constants.TYPE_ID3:
                    result = gson.fromJson(elem, ID3.class);
                    break;
                case Constants.TYPE_DENSITY_CLUSTER:
                    result = gson.fromJson(elem, DensityClustering.class);
                    break;
                case Constants.TYPE_ZERO_R:
                    result = gson.fromJson(elem, ZeroR.class);
                    break;
            }
            if (result != null) result.printClassifierInfo();
            return result;
        }
    }
}
