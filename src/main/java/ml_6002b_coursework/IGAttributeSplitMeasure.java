package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;


public class IGAttributeSplitMeasure extends AttributeSplitMeasure {

    private boolean useGain;

    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        Instances[] instArr;
        int[][] arr;
        //System.out.println(att.isNominal());
        if (att.isNumeric()){
            instArr = splitDataOnNumeric(data, att);
            arr = new int[2][data.numClasses()];
        }else{
            instArr = splitData(data, att);
            arr = new int[att.numValues()][data.numClasses()];
        }

        for (int i = 0; i < instArr.length; i++){
            for (int j = 0; j < instArr[i].size(); j++){
                arr[i][(int) instArr[i].instance(j).classValue()]++;
            }
        }
        if (!useGain){//only use gain
            return AttributeMeasures.measureInformationGain(arr);
        }else{//use gain ratio
            return AttributeMeasures.measureInformationGainRatio(arr);
        }
    }

    public IGAttributeSplitMeasure(){
        useGain = false;
    }
    public IGAttributeSplitMeasure(boolean useGain){
        this.useGain = useGain;
    }

    public void setUseGain(boolean useGain){
        this.useGain = useGain;
    }
    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) throws Exception {
        FileReader fileReader = new FileReader("src/main/java/ml_6002b_coursework/test_data/whisky.arff");
        Instances inst = new Instances(fileReader);
        inst.setClassIndex(inst.numAttributes()-1);

        IGAttributeSplitMeasure ig = new IGAttributeSplitMeasure();
        System.out.println("measure Information Gain for attribute Peaty splitting diagnosis = " + ig.computeAttributeQuality(inst, inst.attribute(0)));
        System.out.println("measure Information Gain for attribute Woody splitting diagnosis = " + ig.computeAttributeQuality(inst, inst.attribute(1)));
        System.out.println("measure Information Gain for attribute Sweat splitting diagnosis = " + ig.computeAttributeQuality(inst, inst.attribute(2)));

        IGAttributeSplitMeasure igRatio = new IGAttributeSplitMeasure(true);
        System.out.println("measure Information Gain Ratio for attribute Peaty splitting diagnosis = " + igRatio.computeAttributeQuality(inst, inst.attribute(0)));
        System.out.println("measure Information Gain Ratio for attribute Woody splitting diagnosis = " + igRatio.computeAttributeQuality(inst, inst.attribute(1)));
        System.out.println("measure Information Gain Ratio for attribute Sweat splitting diagnosis = " + igRatio.computeAttributeQuality(inst, inst.attribute(2)));
    }

}
