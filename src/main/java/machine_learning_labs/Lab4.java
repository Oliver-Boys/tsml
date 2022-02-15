package machine_learning_labs;

import experiments.data.DatasetLoading;
import utilities.ClassifierTools;
import utilities.FoldCreator;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class Lab4 {
    static String[] problems={
            "bank",
            "blood",
            "breast-cancer-wisc-diag",
            "breast-tissue",
            "cardiotocography-10clases",
            "conn-bench-sonar-mines-rocks",
            "conn-bench-vowel-deterding",
            "ecoli",
            "glass",
            "hill-valley",
            "image-segmentation",
            "ionosphere",
            "iris",
            "libras",
            "optical",
            "ozone",
            "page-blocks",
            "parkinsons",
            "planning",
            "post-operative",
            "ringnorm",
            "seeds",
            "spambase",
            "statlog-landsat",
            "statlog-vehicle",
            "steel-plates",
            "synthetic-control",
            "twonorm",
            "vertebral-column-3clases",
            "wall-following",
            "waveform-noise",
            "wine-quality-white",
            "yeast"};
    public static void loadAndRun() throws Exception {
        createOutputFile("E:\\MachineLearningData\\UCI_exp1");
        FileWriter fw = new FileWriter("E:\\MachineLearningData\\UCI_exp1.csv");

        fw.write("problem,bAcc,rfAcc,j48E\n");


        Instances train, test;
        String path = "E:\\MachineLearningData\\UCIContinuous\\";
        System.out.println("Number of problems: " + problems.length);
        double meanB = 0, meanRF = 0, meanDiff = 0;
        int count = 0;
        for(String problem : problems){
            train = DatasetLoading.loadData(path+problem+"\\"+problem+"_TRAIN");
            test = DatasetLoading.loadData(path+problem+"\\"+problem+"_TEST");

            Bagging bagging = new Bagging();
            bagging.setNumIterations(500);

            RandomForest randomForest = new RandomForest();
            randomForest.setNumTrees(500);

            J48Ensemble ensemble = new J48Ensemble();


            bagging.buildClassifier(train);
            randomForest.buildClassifier(train);
            ensemble.buildClassifier(train);

            double accBagging = ClassifierTools.accuracy(test, bagging);
            double accRandomForest = ClassifierTools.accuracy(test, randomForest);
            double accJ48Ensemble = ClassifierTools.accuracy(test, ensemble);

            meanDiff += accRandomForest-accBagging;
            meanB += accBagging;
            meanRF += accRandomForest;

            fw.write(problem+","+accBagging+","+accRandomForest+","+accJ48Ensemble+"\n");

            if(accRandomForest>accBagging)
                count++;
            System.out.println(problem+ " Bagging = "+accBagging+" RandF  = "+accRandomForest);
        }
        fw.close();
        meanDiff/=problems.length;
        meanB/=problems.length;
        meanRF/=problems.length;
        System.out.println("RF wins "+count++);
        System.out.println("Mean Diff "+meanDiff);
        System.out.println("Mean B "+meanB);
        System.out.println("Mean RF  "+meanRF);
    }
    public static void rfEval() throws Exception {
        createOutputFile("E:\\MachineLearningData\\rf");
        FileWriter fw = new FileWriter("E:\\MachineLearningData\\rf.csv");

        fw.write("trees,acc\n");

        Instances train, test;
        String path = "E:\\MachineLearningData\\UCIContinuous\\";
        //int[] rfTrees = {1,2,5,10,20,30,40,50,75,100,200,300,400,500,750,1000};
        int[] rfTrees = {20000};
        double overallAcc = 0;
        for (int treeCount : rfTrees) {

            long t1 = System.nanoTime();

            overallAcc = 0;
            int count = 0;
            for (String problem : problems) {
                train = DatasetLoading.loadData(path + problem + "\\" + problem + "_TRAIN");
                test = DatasetLoading.loadData(path + problem + "\\" + problem + "_TEST");

                RandomForest rf = new RandomForest();
                rf.setNumTrees(treeCount);
                rf.buildClassifier(train);

                double accRF = ClassifierTools.accuracy(test, rf);
                overallAcc+=accRF;
                System.out.print(count++ + ",");
                long elapsedT = System.nanoTime()-t1;
                long eta = (count == 0) ? 0 : (problems.length - count) * (System.nanoTime() - t1) / count;
                String etaMsms = count == 0 ? "N/A" :
                        String.format("%02d:%02d.%02d", TimeUnit.NANOSECONDS.toMinutes(eta),
                                TimeUnit.NANOSECONDS.toSeconds(eta) % TimeUnit.MINUTES.toSeconds(1),
                                TimeUnit.NANOSECONDS.toMillis(eta) % TimeUnit.SECONDS.toMillis(1));
                String elapsed = String.format("%02d:%02d.%02d", TimeUnit.NANOSECONDS.toMinutes(elapsedT),
                        TimeUnit.NANOSECONDS.toSeconds(elapsedT) % TimeUnit.MINUTES.toSeconds(1),
                        TimeUnit.NANOSECONDS.toMillis(elapsedT) % TimeUnit.SECONDS.toMillis(1));

                System.out.print("\rETA: " + etaMsms + ", elapsed: " + elapsed);

            }
            overallAcc/=problems.length;

            System.out.println("\n"+treeCount+","+overallAcc);
            fw.write(treeCount+","+overallAcc+"\n");

        }
        fw.close();
    }

    public static void createOutputFile(String path) throws IOException {
        path = path.trim();
        File file = new File(path+".csv");
        if (file.createNewFile()){
            System.out.println("File created");
        }else{
            System.out.println("File already exists");
        }
    }

    public static void main(String[] args) throws Exception {
        loadAndRun();
        //rfEval();
    }
}
