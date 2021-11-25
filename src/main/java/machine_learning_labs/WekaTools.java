package machine_learning_labs;

import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class WekaTools {
    public static double accuracy(Classifier c, Instances test) throws Exception {

        int count = 0;

        for(Instance instance : test){
            double predicted = c.classifyInstance(instance);
            if(predicted == instance.classValue()){
                count++;
            }
        }
        return ((double) count / test.numInstances());
    }
    public static Instances loadClassificationData(String fullPath){
        Instances instances = null;
        try{
            FileReader reader = new FileReader(fullPath);
            instances = new Instances(reader);
            instances.setClassIndex(instances.numAttributes()-1);
        }catch (FileNotFoundException e){
            System.out.println("Classification data file could not be found: " + fullPath);
        }catch (IOException e){
            System.out.println("Could not convert data to instances: " + e);
        }

        return instances;
    }

    public static Instances[] splitData(Instances all, double proportionTest){
        all.randomize(new Random());

        Instances[] split = new Instances[2];
        split[0] = new Instances(all);
        split[1] = new Instances(all,0);

        for(int i = 0; i < (all.numInstances()*proportionTest); i++){
            split[1].add(split[0].get(i));
            split[0].remove(i);
        }
        return split;
    }

    public static double[] classDistribution(Instances data){
        double[] classDist = new double[data.numClasses()];

        for(Instance instance : data){
            classDist[(int) instance.classValue()]++;
        }
        for(int i = 0; i < classDist.length; i++){
            classDist[i] /= data.numInstances();
        }
        return classDist;
    }
    /*public static int[][] confusionMatrix(int[] predicted, int[] actual){
        int[][] matrix = new int[2][2];


        for(int i = 0; i < predicted.length; i++){
            System.out.println(predicted[i] + "," + actual[i]);
            if(actual[i]==0){
                if(predicted[i]==0){
                    matrix[0][0]++;
                }else{
                    matrix[0][1]++;
                }
            }else{
                if(predicted[i]==0){
                    matrix[1][0]++;
                }else{
                    matrix[1][1]++;
                }
            }
        }
        return matrix;
    }*/
    public static int[][] confusionMatrix(Classifier c, Instances test){
        int[][] matrix = new int[test.numClasses()][test.numClasses()];
        int[] predicted = classifyInstances(c,test);
        int[] actual = getClassValues(test);
        for(int i = 0; i < predicted.length; i++){
            matrix[predicted[i]][actual[i]]++;
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

    public static void main(String[] args) throws Exception {
        //Instances test = loadClassificationData("C:\\Users\\purpl\\Documents\\Uni\\YEAR 3 PROJECT\\ML Lab Sheets\\Lab 1\\Arsenal_TEST.arff");
        //Instances train = loadClassificationData("C:\\Users\\purpl\\Documents\\Uni\\YEAR 3 PROJECT\\ML Lab Sheets\\Lab 1\\Arsenal_TRAIN.arff");

        /*int seed = 0;
        Instances[] trainTest = DatasetLoading.sampleItalyPowerDemand(seed);
        Instances train1 = trainTest[0];
        Instances test1 = trainTest[1];*/

        Instances data = loadClassificationData("E:\\MachineLearningData\\Aedes_Female_VS_House_Fly_POWER.arff");



        ArrayList<Classifier> classifiers = new ArrayList<>();
        classifiers.add(new IB1());
        classifiers.add(new IBk());
        classifiers.add(new J48());
        classifiers.add(new Logistic());
        for(int i = 0; i < 30; i++) {
            System.out.println("Run " + (i+1));
            Instances[] trainTest = splitData(data, 0.3);
            for (Classifier classifier : classifiers) {
                try {
                    classifier.buildClassifier(trainTest[0]);
                    System.out.print(accuracy(classifier, trainTest[1]) + ",");
                    //System.out.println(accuracy(classifier, trainTest[1]));
                    //System.out.println(Arrays.deepToString(confusionMatrix(classifier, trainTest[1])));
                } catch (Exception e) {
                    System.out.println(e);
                }
            }
            System.out.print("\n");
        }



    }
}
