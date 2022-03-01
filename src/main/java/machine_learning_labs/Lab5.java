package machine_learning_labs;

import evaluation.storage.ClassifierResults;
import experiments.CollateResults;
import experiments.data.DatasetLoading;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.IOException;

public class Lab5 {
    public static void main(String[] args) throws Exception {
        /*calcResults(new NaiveBayes(), "E:\\MachineLearningData\\", "JW_RedVsBlack0");

        ClassifierResults results = new ClassifierResults();
        results.loadResultsFromFile("E:\\MachineLearningData\\results\\lab5_res1.csv");
        results.findAllStats();*/

        calcCompare(new NaiveBayes(), new BayesNet(), "E:\\MachineLearningData\\UCIContinuous\\", "bank" , 1);

        String[] str = {"E:\\MachineLearningData\\results\\","E:\\MachineLearningData\\UCIContinuous\\","1","false","BayesNet","0"};
        CollateResults.collate(str);
    }

    public static void calcCompare(Classifier c1, Classifier c2, String dataLoc, String datasetname, int numFolds) throws Exception {
        Instances instances = DatasetLoading.loadData(dataLoc + datasetname + "\\" + datasetname);

        for(int i = 0; i < numFolds; i++) {
            ClassifierResults r1 = new ClassifierResults();
            ClassifierResults r2 = new ClassifierResults();

            r1.setClassifierName(c1.getClass().getSimpleName());
            r1.setDatasetName(datasetname);
            r2.setClassifierName(c2.getClass().getSimpleName());
            r2.setDatasetName(datasetname);

            Instances[] trainTest = WekaTools.splitData(instances, 0.3);

            c1.buildClassifier(trainTest[0]);
            c2.buildClassifier(trainTest[0]);

            for (Instance test : trainTest[1]){
                long t1 = System.nanoTime();
                double[] distribution = c1.distributionForInstance(test);
                long t2 = System.nanoTime() - t1;
                double predicted = max(distribution);

                r1.addPrediction(test.classValue(), distribution, predicted, t2, "");

                t1 = System.nanoTime();
                distribution = c1.distributionForInstance(test);
                t2 = System.nanoTime() - t1;
                predicted = max(distribution);

                r2.addPrediction(test.classValue(), distribution, predicted, t2, "");
            }

            File file = new File("E:\\MachineLearningData\\results\\"+ r1.getClassifierName() + "\\predictions\\" + r1.getDatasetName() + "\\");
            if (!file.exists())
                file.mkdirs();
            file = new File("E:\\MachineLearningData\\results\\"+ r2.getClassifierName() + "\\predictions\\" + r2.getDatasetName() + "\\");
            if (!file.exists())
                file.mkdirs();

            r1.finaliseResults();
            r1.findAllStats();
            r1.writeFullResultsToFile("E:\\MachineLearningData\\results\\" + r1.getClassifierName() + "\\predictions\\" + r1.getDatasetName() + "\\testFold" + i + ".csv");

            r2.finaliseResults();
            r2.findAllStats();
            r2.writeFullResultsToFile("E:\\MachineLearningData\\results\\" + r2.getClassifierName() + "\\predictions\\" + r2.getDatasetName() + "\\testFold" + i + ".csv");

        }

    }

    public static void calcResults(Classifier c, String dataLoc, String datasetName) throws Exception {
        ClassifierResults results = new ClassifierResults();

        results.setClassifierName(c.getClass().getSimpleName());
        results.setDatasetName(datasetName);

        Instances train = DatasetLoading.loadData(dataLoc + datasetName + "_TRAIN");
        Instances test = DatasetLoading.loadData(dataLoc + datasetName + "_TEST");

        c.buildClassifier(train);

        for (Instance instance : test){
            long t1 = System.nanoTime();
            double[] distribution = c.distributionForInstance(instance);
            long t2 = System.nanoTime() - t1;

            double predicted = max(distribution);

            results.addPrediction(instance.classValue(), distribution, predicted, t2, "");
        }
        results.finaliseResults();
        results.findAllStats();
        results.writeFullResultsToFile("E:\\MachineLearningData\\results\\lab5_res1.csv");
    }
    private static double max(double[] ar){
        double max = 0;
        double maxPos = -1;
        for (int i = 0; i< ar.length; i++){
            if (ar[i] > max){
                max = ar[i];
                maxPos = i;
            }
        }
        return maxPos;
    }
}
