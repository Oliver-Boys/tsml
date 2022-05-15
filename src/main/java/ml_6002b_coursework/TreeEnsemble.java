package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.RandomSubset;
import java.io.FileReader;
import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {

    CourseworkTree[] trees;
    int numTrees = 50;
    boolean averageDistributions = false;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        Random random = new Random();
        String[] split = new String[]{"ig","igratio","chi","gini"};
        trees = new CourseworkTree[numTrees];
        RandomSubset filter = new RandomSubset();
        Instances instToFilter = data;
        for (int i = 0; i < trees.length; i++){
            //Random subset of attributes
            trees[i] = new CourseworkTree();
            int seed = random.nextInt();
            filter.setSeed(seed);
            filter.setNumAttributes(0.5);// 50% of attributes
            filter.setInputFormat(instToFilter);
            //System.out.println(data);
            Instances inst = filter.process(instToFilter);
            //System.out.println(inst);

            //Randomise split criterion for each tree
            int rand = Math.abs(random.nextInt());
            System.out.println(rand);
            int select = rand%4;
            //System.out.println(select);
            String[] options = new String[]{"-split",split[select]};
            trees[i].setOptions(options);
            instToFilter = inst;
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

    public static void main(String[] args) throws Exception {
        FileReader optdigits = new FileReader("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");
        FileReader chinatown = new FileReader("src/main/java/ml_6002b_coursework/test_data/Chinatown.arff");


        Instances inst = new Instances(optdigits);
        inst.setClassIndex(inst.numAttributes()-1);
        TreeEnsemble ensemble = new TreeEnsemble();
        ensemble.buildClassifier(inst);
    }
}
