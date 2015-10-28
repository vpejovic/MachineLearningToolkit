package si.uni_lj.fri.lrss.machinelearningtoolkit.utils;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by veljko on 04/09/15.
 */
public class FeatureNominal extends Feature {

    protected ArrayList<String> mCategories;
    protected HashMap<String,Integer> mCategoryIndex;

    /**
     *
     * Create a feature object with the given name and type.
     * @param fname Feature name.
     * @param fvalues
     * @throws MLException
     */
    public FeatureNominal(String fname, ArrayList<String> fvalues) {
        super(fname, NOMINAL);
        mCategories = (ArrayList<String>) fvalues.clone();
        mCategoryIndex = new HashMap<String, Integer>();
        for(int i=0;i<fvalues.size();i++) mCategoryIndex.put(fvalues.get(i), i);
    }

    public ArrayList<String> getValues(){
        return mCategories;
    }

    public int indexOfCategory(String value){
        //Log.d(TAG, "feature "+mName+" going for value "+value);
        return mCategoryIndex.get(value);

    }

    public String categoryOfIndex(int index){
        return mCategories.get(index);
    }

    public int numberOfCategories(){
        if (mType == NOMINAL) return mCategories.size();
        else return 1;
    }
}
