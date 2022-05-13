package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instances;

import java.io.FileReader;

public class ChiSquaredAttributeSplitMeasure extends AttributeSplitMeasure{
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        Instances[] instArr = splitData(data, att);
        int[][] arr = new int[att.numValues()][data.numClasses()];
        for (int i = 0; i < instArr.length; i++){
            for (int j = 0; j < instArr[i].size(); j++){
                arr[i][(int) instArr[i].instance(j).classValue()]++;
            }
        }
        return AttributeMeasures.measureChiSquared(arr);
    }

    public static void main(String[] args) throws Exception {
        FileReader fileReader = new FileReader("src/main/java/ml_6002b_coursework/test_data/whisky.arff");
        Instances inst = new Instances(fileReader);
        inst.setClassIndex(inst.numAttributes()-1);

        ChiSquaredAttributeSplitMeasure chi = new ChiSquaredAttributeSplitMeasure();
        System.out.println("measure Chi Squared for attribute Peaty splitting diagnosis = " + chi.computeAttributeQuality(inst, inst.attribute(0)));
        System.out.println("measure Chi Squared for attribute Woody splitting diagnosis = " + chi.computeAttributeQuality(inst, inst.attribute(1)));
        System.out.println("measure Chi Squared for attribute Sweat splitting diagnosis = " + chi.computeAttributeQuality(inst, inst.attribute(2)));
    }
}
