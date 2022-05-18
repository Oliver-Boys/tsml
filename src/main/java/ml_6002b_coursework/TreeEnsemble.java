package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WekaEnumeration;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RandomSubset;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {

    CourseworkTree[] trees;
    int[] seeds;
    int numTrees = 2;
    boolean averageDistributions = false;
    double attributeSubSamplePercent = 0.5;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        Random random = new Random();
        String[] split = new String[]{"ig","igratio","chi","gini"};
        trees = new CourseworkTree[numTrees];
        seeds = new int[numTrees];
        RandomSubset filter = new RandomSubset();
        filter.setNumAttributes(attributeSubSamplePercent*data.numAttributes());// 50% of attributes

        //filter.setOptions(new String[]{"-V"});
        //Instances instToFilter = data;
        for (int i = 0; i < trees.length; i++){
            //System.out.println("\t " + i);
            //Random subset of attributes
            trees[i] = new CourseworkTree();
            int seed = random.nextInt();
            seeds[i] = seed;
            filter.setSeed(seed);
            filter.setInputFormat(data);
            //System.out.println(data);
            Instances inst = filter.process(data);

            /*Enumeration enumeration = inst.enumerateAttributes();
            while (enumeration.hasMoreElements()){
                System.out.println(enumeration.nextElement());
            }*/

            //System.out.println(inst);


            //Randomise split criterion for each tree
            int rand = Math.abs(random.nextInt());
            //System.out.println(rand);
            int select = rand%4;
            //System.out.println(select);
            String[] options = new String[]{"-split",split[select]};
            trees[i].setOptions(options);
            //instToFilter = inst;

            trees[i].buildClassifier(inst);

        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        RandomSubset filter = new RandomSubset();
        filter.setNumAttributes(attributeSubSamplePercent*instance.numAttributes());// 50% of attributes

        //Instances instances = new Instances(null);


        if (!averageDistributions){
            int[] classArr = new int[instance.numClasses()];
            //System.out.print("\t\tclassify");
            for (int i = 0; i < trees.length; i++){
                filter.setSeed(seeds[i]);
                filter.setInputFormat(instance.dataset());
                filter.input(instance);
                Instance inst = filter.output();

                double predicted = trees[i].classifyInstance(inst);
                classArr[(int) predicted]++;
            }
            int highestCount = 0;
            double highestClass = -1;
            for (int i = 0; i < classArr.length; i++){
                if (classArr[i] > highestCount){
                    highestClass = i;
                }
            }
            return highestClass;
        }else{
            double[] dist = new double[instance.numClasses()];
            for (int i = 0; i < trees.length; i++){

                filter.setSeed(seeds[i]);
                filter.setInputFormat(instance.dataset());
                filter.input(instance);
                Instance inst = filter.output();

                double[] treeDist = trees[i].distributionForInstance(instance);
                for (int j = 0; j < treeDist.length; j++){
                    dist[j] += treeDist[j];
                    dist[j] /= 2;
                }
            }
            int highestProb = 0;
            double highestClass = -1;
            for (int i = 0; i < dist.length; i++){
                if (dist[i] > highestProb){
                    highestClass = i;
                }
            }
            return highestClass;
        }

    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        RandomSubset filter = new RandomSubset();
        filter.setNumAttributes(attributeSubSamplePercent*instance.numAttributes());// 50% of attributes
        if (!averageDistributions){
            double[] dist = new double[instance.numClasses()];
            for (int i = 0; i < trees.length; i++){
                filter.setSeed(seeds[i]);
                filter.setInputFormat(instance.dataset());
                filter.input(instance);
                Instance inst = filter.output();

                dist[(int) trees[i].classifyInstance(inst)]++;
            }
            for (int i = 0; i < dist.length; i++){
                dist[i] /= numTrees;
            }
            return dist;
        }else{
            double[] dist = new double[instance.numClasses()];
            for (int j = 0; j < trees.length; j++){
                filter.setSeed(seeds[j]);
                filter.setInputFormat(instance.dataset());
                filter.input(instance);
                Instance inst = filter.output();

                double[] treeDist = trees[j].distributionForInstance(inst);
                for (int i = 0; i < treeDist.length; i++){
                    dist[i] += treeDist[i];
                    dist[i] /= 2;
                }
            }
            return dist;
        }
    }

    public static void main(String[] args) throws Exception {
        FileReader optdigits = new FileReader("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");
        FileReader chinatown = new FileReader("src/main/java/ml_6002b_coursework/test_data/Chinatown.arff");
        Instances instOptdigit = new Instances(optdigits);
        Instances instChinatown = new Instances(chinatown);

        Instances[] trainTest = Utils.splitData(instOptdigit, 0.5);
        trainTest[0].setClassIndex(trainTest[0].numAttributes()-1);
        trainTest[1].setClassIndex(trainTest[1].numAttributes()-1);


        TreeEnsemble ensemble = new TreeEnsemble();
        ensemble.buildClassifier(trainTest[0]);
        System.out.println("Acc for optdigits: " + ClassifierTools.accuracy(trainTest[1], ensemble));
        for (int i = 0; i < 5; i++){
            System.out.println("Acc for optdigits test case " + i + ": " + ensemble.classifyInstance(trainTest[1].get(i)));
        }


        trainTest = Utils.splitData(instChinatown, 0.5);
        trainTest[0].setClassIndex(trainTest[0].numAttributes()-1);
        trainTest[1].setClassIndex(trainTest[1].numAttributes()-1);

        ensemble = new TreeEnsemble();
        ensemble.buildClassifier(trainTest[0]);
        System.out.println("Acc for Chinatown: " + ClassifierTools.accuracy(trainTest[1], ensemble));
        for (int i = 0; i < 5; i++){
            System.out.println("Acc for Chinatown test case " + i + ": " + ensemble.classifyInstance(trainTest[1].get(i)));
        }

    }
}
