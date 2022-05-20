package ml_6002b_coursework;

import weka.classifiers.Classifier;
import weka.core.Instances;
import java.util.Random;

public class Utils {
    public static Instances[] splitData(Instances all, double proportionTest){
        all.randomize(new Random());

        Instances[] split = new Instances[2];
        split[0] = new Instances(all);
        split[1] = new Instances(all,0);
        //System.out.println(all.numInstances());
        for(int i = 0; i < (all.numInstances()*proportionTest); i++){
            split[1].add(split[0].get(0));
            split[0].remove(0);
            //System.out.println(split[1].size() + "," + split[0].size());
        }
        //System.out.println("done");
        return split;
    }
    public static int[][] confusionMatrix(Classifier c, Instances test){
        int[][] matrix = new int[test.numClasses()][test.numClasses()];
        int[] predicted = classifyInstances(c,test);
        int[] actual = getClassValues(test);
        for(int i = 0; i < predicted.length; i++){
            matrix[actual[i]][predicted[i]]++;
        }
        return matrix;
    }
    public static int[] classifyInstances(Classifier c, Instances test) {
        int[] predicted = new int[test.numInstances()];
        try {
            for (int i = 0; i < predicted.length; i++) {
                predicted[i] = (int) c.classifyInstance(test.instance(i));
            }
        }catch (Exception e){
            System.out.println("Exception " + e);
        }
        return predicted;
    }
    public static int[] getClassValues(Instances data){
        int[] actual = new int[data.numInstances()];
        for (int i = 0; i < actual.length; i++) {
            actual[i] = (int) data.get(i).classValue();
        }
        return actual;
    }
}
