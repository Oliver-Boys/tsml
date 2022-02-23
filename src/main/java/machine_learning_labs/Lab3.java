package machine_learning_labs;

import experiments.data.DatasetLoading;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.InfoGainSplitCrit;
import weka.core.Instances;

import java.io.IOException;

public class Lab3 {
    public static void golfExample() throws Exception {
        Instances instances = DatasetLoading.loadData("E:\\MachineLearningData\\golf");
        Distribution d = new Distribution(instances);

        System.out.println(d.dumpDistribution());
    }
    public static void main(String[] args) throws Exception {
        golfExample();
    }
}
