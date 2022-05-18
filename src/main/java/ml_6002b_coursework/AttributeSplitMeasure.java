package ml_6002b_coursework;

import org.checkerframework.checker.units.qual.A;
import org.w3c.dom.Attr;
import scala.Int;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

/**
 * Interface for alternative attribute split measures for Part 2.2 of the coursework
 */
public abstract class AttributeSplitMeasure {

    public abstract double computeAttributeQuality(Instances data, Attribute att) throws Exception;

    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
    public Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int i = 0; i < att.numValues(); i++) {
            splitData[i] = new Instances(data, data.numInstances());
        }

        for (Instance inst: data) {
            splitData[(int) inst.value(att.index())].add(inst);
        }

        for (Instances split : splitData) {
            split.compactify();
        }

        return splitData;
    }
    public Instances[] splitDataOnNumeric(Instances data, Attribute att){
        Instances[] splitData = new Instances[2];
        splitData[0] = new Instances(data,0);
        splitData[1] = new Instances(data, 0);
        //System.out.println(splitData[0]);
        AttributeStats stats = data.attributeStats(att.index());
        double min = stats.numericStats.min;
        double max = stats.numericStats.max;
        double mean = stats.numericStats.mean;
        //System.out.println("mean" + mean);
        for(int i = 0; i < data.numInstances(); i++){
            //System.out.println(i);
            //System.out.println(data.instance(i).value(att));
            if (data.instance(i).value(att) <= mean){
                splitData[0].add(data.instance(i));
            }else{
                splitData[1].add(data.instance(i));
            }
        }
        //System.out.println(splitData[0].size());
        //System.out.println(splitData[1].size());
        return splitData;
    }

}
