package machine_learning_labs;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

public class HistogramClassifier implements Classifier {

    int numberOfBins;
    int attributePos;
    int[][] histograms;
    double interval;
    double min;
    double max;

    public HistogramClassifier(){
        numberOfBins = 10;
        attributePos = 0;
    }

    public void setNumberOfBins(int numberOfBins){
        this.numberOfBins = numberOfBins;
    }
    public void setAttributePosition(int attributePosition){
        this.attributePos = attributePosition;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        int numClasses = instances.numClasses();
        histograms = new int[numClasses][numberOfBins];

        min = instances.attributeStats(attributePos).numericStats.min;
        max = instances.attributeStats(attributePos).numericStats.max;
        interval = (max-min)/ numberOfBins;

        for(Instance instance : instances){
            int pos = (int)((instance.value(attributePos) - min)/interval);
            histograms[(int)instance.classValue()][(pos == numberOfBins)?numberOfBins-1:pos]++;
        }

        System.out.println(Arrays.deepToString(histograms));
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] predicted = distributionForInstance(instance);

        double maxClass = 0.0f;
        double maxProbability = 0.0f;

        for(int i = 0; i < predicted.length; i++){
            if(predicted[i]>maxProbability){
                maxClass = i;
                maxProbability = predicted[i];
            }
        }
        return maxClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        int numClasses = histograms.length;
        double[] probability = new double[numClasses];
        int[] frequenciesPerClass = new int[numClasses];
        int count = 0;

        int pos = (int)((instance.value(attributePos) - min)/interval);
        for(int i = 0; i < numClasses; i++){
            frequenciesPerClass[i] = histograms[i][pos];
            count += histograms[i][pos];
        }

        if(count==0){
            Arrays.fill(frequenciesPerClass,1);
            count = numClasses;
        }
        for(int i = 0; i < numClasses; i++){
            probability[i] = (double)frequenciesPerClass[i]/count;
        }

        return probability;
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}
