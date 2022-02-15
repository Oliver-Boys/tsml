package machine_learning_labs;

import org.checkerframework.checker.units.qual.A;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

public class J48Ensemble extends AbstractClassifier {
    private ArrayList<Classifier> j48s;
    int numClassifiers = 100;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        j48s = new ArrayList<>();
        for (int i = 0; i < numClassifiers; i++){
            Classifier c = new J48();
            j48s.add(c);
        }
        for (Classifier c : j48s){
            data.randomize(new Random());
            Instances train = new Instances(data,0,data.size()/2);
            c.buildClassifier(train);
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        int[] classCount = new int[instance.numClasses()];
        for (Classifier c : j48s){
            classCount[(int)c.classifyInstance(instance)]++;
        }
        int maxCount = 0, maxClass = 0;
        for (int i = 0; i < classCount.length; i++){
            if (classCount[i]>maxCount){
                maxCount = classCount[i];
                maxClass = i;
            }
        }
        return maxClass;
    }
}
