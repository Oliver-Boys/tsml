package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import tsml.classifiers.distance_based.utils.collections.tree.Tree;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.lazy.IB1;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.ADTree;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

public class Experiment {
    public static void main(String[] args) throws Exception {
        String[] datasets = DatasetLists.nominalAttributeProblems;
        Classifier[] classifiers = new Classifier[2];
        classifiers[0] = new Id3();
        classifiers[1] = new J48();
        double[][] accuracy = new double[datasets.length][8];
        double[][] ram = new double[datasets.length][8];
        double[][] timeTrain = new double[datasets.length][8];
        double[][] timeTest = new double[datasets.length][8];

        /*String[] selectionMethods = new String[]{"ig","igratio","chi","gini"};
        CourseworkTree tree = new CourseworkTree();
        for (int k = 0; k < 30; k++) {
            System.out.println(k);
            for (int i = 0; i < datasets.length; i++) {
                Instances inst = DatasetLoading.loadData("E:\\MachineLearningData\\UCI Discrete\\UCI Discrete\\" + datasets[i] + "\\" + datasets[i]);
                Instances[] trainTest = Utils.splitData(inst, 0.5);
                trainTest[0].setClassIndex(trainTest[0].numAttributes() - 1);
                trainTest[1].setClassIndex(trainTest[1].numAttributes() - 1);
                for (int j = 0; j < selectionMethods.length; j++) {
                    tree.setOptions(new String[]{"-split", selectionMethods[j]});
                    tree.buildClassifier(trainTest[0]);
                    double acc = ClassifierTools.accuracy(trainTest[1], tree);
                    //System.out.println(i + "," + j);
                    accuracy[i][j] += acc;
                    if (k >= 1){
                        accuracy[i][j] /=2;
                    }
                }
            }
        }*/

        String[] selectionMethods = new String[]{"ig","igratio","chi","gini"};

        AbstractClassifier[] classifiers1 = new AbstractClassifier[8];

        String[] names = new String[]{"random_forest","rotation_forest","naive_bayes","bayes_net","ib1","j48","cw_ensemble_50","cw_ensemble_100"};

        classifiers1[0] = new RandomForest();
        classifiers1[1] = new RotationForest();
        classifiers1[2] = new NaiveBayes();
        classifiers1[3] = new BayesNet();
        classifiers1[4] = new IB1();
        classifiers1[5] = new J48();
        classifiers1[6] = new TreeEnsemble();
        classifiers1[7] = new TreeEnsemble();

        //Discretize filter = new Discretize();

        classifiers1[6].setOptions(new String[]{"-S","0.5"});
        classifiers1[7].setOptions(new String[]{"-S","1"});
        for (int k = 0; k < 500; k++){
            System.out.println(k);
            for (int i = 0; i < 1; i++){
                System.out.println(i);
                //Instances inst = DatasetLoading.loadData("E:\\MachineLearningData\\UCI Discrete\\UCI Discrete\\" + datasets[i] + "\\" + datasets[i]);
                Instances train = DatasetLoading.loadData("E:\\MachineLearningData\\Car\\Car_TRAIN");
                Instances test = DatasetLoading.loadData("E:\\MachineLearningData\\Car\\Car_TEST");

                //filter.setInputFormat(train);

                //Instances trainFiltered = Filter.useFilter(train, filter);
                //Instances testFiltered = Filter.useFilter(test, filter);

                //Instances[] trainTest = Utils.splitData(inst, 0.5);
                //trainTest[0].setClassIndex(trainTest[0].numAttributes() - 1);
                //trainTest[1].setClassIndex(trainTest[1].numAttributes() - 1);
                for (int j = 0; j < classifiers1.length; j++) {
                    //tree.setOptions(new String[]{"-split", selectionMethods[j]});
                    System.gc();
                    long t1 = System.nanoTime();
                    long mem1 = ((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())/1024);
                    classifiers1[j].buildClassifier(train);
                    long mem2 = ((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())/1024);
                    long t2 = System.nanoTime();
                    timeTrain[i][j] += (t2-t1);
                    ram[i][j] += (mem2-mem1);
                    System.out.println("built");
                    t1 = System.nanoTime();
                    double acc = ClassifierTools.accuracy(test, classifiers1[j]);
                    t2 = System.nanoTime();
                    timeTest[i][j] += (t2-t1);
                    //System.out.println(i + "," + j);
                    accuracy[i][j] += acc;
                    if (k >= 1){
                        accuracy[i][j] /=2;
                        timeTrain[i][j] /=2;
                        timeTest[i][j] /=2;
                        ram[i][j] /=2;
                    }

                    System.out.println(classifiers1[j].getClass().getName() + " | " + Arrays.deepToString(Utils.confusionMatrix(classifiers1[j], test)));
                }

                /*for (int j = 0; j < classifiers.length; j++){
                    classifiers[j].buildClassifier(trainTest[0]);
                    double acc = ClassifierTools.accuracy(trainTest[1], classifiers[j]);
                    accuracy[i][j+4] += acc;
                    if (k >= 1){
                        accuracy[i][j+4] /=2;
                    }
                }*/
            }
        }
        for (int i = 0; i < classifiers1.length; i++) {
            File file = new File("E:\\MachineLearningData\\results\\cw\\caseStudy\\result_" + names[i] + ".csv");
            file.createNewFile();
            FileWriter fw = new FileWriter(file);
            for (int j = 0; j < 1; j++) {
                fw.write("Car" + "," + accuracy[j][i] + "," + timeTrain[j][i] + "," + timeTest[j][i] + "," + ram[j][i] + "\n");
            }
            fw.close();
        }


    }
}
