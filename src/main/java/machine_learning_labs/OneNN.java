package machine_learning_labs;

import experiments.data.DatasetLoading;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;

public class OneNN extends AbstractClassifier {
    Instances trainData;
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainData = data;
    }

    @Override
    public double classifyInstance(Instance instance){
        double minRange = Double.MAX_EXPONENT;
        double minClass = 0;
        for (Instance i1 : trainData){
            double test = dist(i1,instance);
            if (test < minRange){
                minRange = test;
                minClass = i1.classValue();
            }
        }
        return minClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] prediction = new double[instance.numClasses()];
        double classVal = classifyInstance(instance);
        for (int i = 0; i < prediction.length; i++){
            if (i != classVal){
                prediction[i] = 0.0;
            }else{
                prediction[i] = 1.0;
            }
        }
        return prediction;
    }

    private double dist(Instance i1, Instance i2){
        int len = i1.numAttributes();
        int classPos = i1.classIndex();

        int sum = 0;
        for (int i = 0; i < len; i++){
            if (i != classPos || i1.attribute(i).isNumeric())
                sum += Math.pow(i1.value(i) - i2.value(i),2);
        }
        return Math.sqrt(sum);
    }

    public static void main(String[] args) throws Exception {
        /*Instances train = DatasetLoading.loadData("E:\\MachineLearningData\\Arsenal_TRAIN");
        Instances test = DatasetLoading.loadData("E:\\MachineLearningData\\Arsenal_TEST");

        Classifier oneNN = new OneNN();
        oneNN.buildClassifier(train);
        double acc = ClassifierTools.accuracy(test, oneNN);
        System.out.println("Accuracy: " + acc);*/

        Instances football = DatasetLoading.loadData("E:\\MachineLearningData\\FootballPlayers");

        Instances[] trainTest = WekaTools.splitData(football,0.3);
        Classifier oneNN = new OneNN();
        oneNN.buildClassifier(trainTest[0]);
        double acc = ClassifierTools.accuracy(trainTest[1], oneNN);
        System.out.println("Acc Mine: " + acc);

        Classifier[] classifiers = {new IB1(), new IBk(), new MultilayerPerceptron(), new SMO()};
        for (Classifier c : classifiers){
            c.buildClassifier(trainTest[0]);
            acc = ClassifierTools.accuracy(trainTest[1], c);
            System.out.println("Acc " + c.getClass().getSimpleName() + ": " + acc);
        }
    }
}
