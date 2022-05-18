package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import tsml.classifiers.distance_based.utils.collections.tree.Tree;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class Experiment {
    public static void main(String[] args) throws Exception {
        String[] datasets = DatasetLists.nominalAttributeProblems;
        Classifier[] classifiers = new Classifier[2];
        classifiers[0] = new Id3();
        classifiers[1] = new J48();
        double[][] accuracy = new double[datasets.length][1];
        double[][] ram = new double[datasets.length][1];
        double[][] timeTrain = new double[datasets.length][1];
        double[][] timeTest = new double[datasets.length][1];

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
        TreeEnsemble tree = new TreeEnsemble();
        for (int k = 0; k < 1; k++){
            System.out.println(k);
            for (int i = 0; i < datasets.length; i++){
                System.out.println(i);
                Instances inst = DatasetLoading.loadData("E:\\MachineLearningData\\UCI Discrete\\UCI Discrete\\" + datasets[i] + "\\" + datasets[i]);
                Instances[] trainTest = Utils.splitData(inst, 0.5);
                trainTest[0].setClassIndex(trainTest[0].numAttributes() - 1);
                trainTest[1].setClassIndex(trainTest[1].numAttributes() - 1);
                for (int j = 0; j < 1; j++) {
                    //tree.setOptions(new String[]{"-split", selectionMethods[j]});
                    long t1 = System.nanoTime();
                    tree.buildClassifier(trainTest[0]);
                    long t2 = System.nanoTime();
                    timeTrain[i][j] += (t2-t1);
                    System.out.println("built");
                    t1 = System.nanoTime();
                    double acc = ClassifierTools.accuracy(trainTest[1], tree);
                    t2 = System.nanoTime();
                    timeTest[i][j] += (t2-t1);
                    //System.out.println(i + "," + j);
                    accuracy[i][j] += acc;
                    if (k >= 1){
                        accuracy[i][j] /=2;
                        timeTrain[i][j] /=2;
                        timeTest[i][j] /=2;
                    }
                }
                tree = new TreeEnsemble();
                /*for (int j = 0; j < classifiers.length; j++){
                    classifiers[j].buildClassifier(trainTest[0]);
                    double acc = ClassifierTools.accuracy(trainTest[1], classifiers[j]);
                    accuracy[i][j+4] += acc;
                    if (k >= 1){
                        accuracy[i][j+4] /=2;
                    }
                }*/
            }
        }File file = new File("E:\\MachineLearningData\\results\\cw\\result_tune_ensemble_50%_2_tree.csv");
        file.createNewFile();
        FileWriter fw = new FileWriter(file);
        for (int i = 0; i < datasets.length; i++){
            fw.write(datasets[i] + "," + accuracy[i][0] + "\n");
        }
        fw.close();


    }
}
