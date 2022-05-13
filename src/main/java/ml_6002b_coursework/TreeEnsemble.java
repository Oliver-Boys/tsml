package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {

    CourseworkTree[] trees;
    int numTrees = 50;
    boolean averageDistributions = false;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        trees = new CourseworkTree[50];




        //Randomise split criterion for each tree
        Random random = new Random();
        String[] split = new String[]{"ig","igratio","chi","gini"};
        for (CourseworkTree tree : trees){
            int rand = random.nextInt();
            int select = rand%4;
            tree.setOptions(new String[]{"-split",split[select]});
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (!averageDistributions){
            int[] classArr = new int[instance.numClasses()];
            for (CourseworkTree tree : trees){
                double predicted = tree.classifyInstance(instance);
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
            for (CourseworkTree tree : trees){
                double[] treeDist = tree.distributionForInstance(instance);
                for (int i = 0; i < treeDist.length; i++){
                    dist[i] += treeDist[i];
                    dist[i] /= 2;
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
        if (!averageDistributions){
            double[] dist = new double[instance.numClasses()];
            for (CourseworkTree tree: trees){
                dist[(int) tree.classifyInstance(instance)]++;
            }
            for (int i = 0; i < dist.length; i++){
                dist[i] /= numTrees;
            }
            return dist;
        }else{
            double[] dist = new double[instance.numClasses()];
            for (CourseworkTree tree : trees){
                double[] treeDist = tree.distributionForInstance(instance);
                for (int i = 0; i < treeDist.length; i++){
                    dist[i] += treeDist[i];
                    dist[i] /= 2;
                }
            }
            return dist;
        }
    }
}
