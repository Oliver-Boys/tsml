package machine_learning_labs;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class MajorityClass extends AbstractClassifier {

    double mostOccuringClass;

    public MajorityClass(){
        mostOccuringClass = -1;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        double[] classDist = classDistribution(data);
        mostOccuringClass = -1;
        double max = -1, prob = -1;

        for(int i = 0; i < classDist.length; i++){
            if(classDist[i]>0.5){
                mostOccuringClass = i;
                break;
            }
            if(classDist[i]>prob){
                prob = classDist[i];
                max = i;
            }
        }
        if(mostOccuringClass == -1)
            mostOccuringClass = max;

    }

    @Override
    public double classifyInstance(Instance data){
        return mostOccuringClass;
    }

    private static double[] classDistribution(Instances data){
        double[] classDist = new double[data.numClasses()];

        for(Instance instance : data){
            classDist[(int) instance.classValue()]++;
        }
        for(int i = 0; i < classDist.length; i++){
            classDist[i] /= data.numInstances();
        }
        return classDist;
    }
}
