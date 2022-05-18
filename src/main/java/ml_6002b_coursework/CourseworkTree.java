package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import org.checkerframework.checker.units.qual.C;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.io.FileReader;
import java.util.Arrays;

/**
 * A basic decision tree classifier for use in machine learning coursework (6002B).
 */
public class CourseworkTree extends AbstractClassifier {

    /** Measure to use when selecting an attribute to split the data with. */
    private AttributeSplitMeasure attSplitMeasure = new IGAttributeSplitMeasure();

    /** Maxiumum depth for the tree. */
    private int maxDepth = Integer.MAX_VALUE;

    /** The root node of the tree. */
    private TreeNode root;

    private double mean;

    /**
     * Sets the attribute split measure for the classifier.
     *
     * @param attSplitMeasure the split measure
     */
    public void setAttSplitMeasure(AttributeSplitMeasure attSplitMeasure) {
        this.attSplitMeasure = attSplitMeasure;
    }

    /**
     * Sets the max depth for the classifier.
     *
     * @param maxDepth the max depth
     */
    public void setMaxDepth(int maxDepth){
        this.maxDepth = maxDepth;
    }


    @Override
    public void setOptions(String[] options) throws Exception {
        //System.out.println(Arrays.toString(options));
        String s = weka.core.Utils.getOption("split", options).toLowerCase();
        if (s.equals("ig")){
            attSplitMeasure = new IGAttributeSplitMeasure(false);
        }else if (s.equals("igratio")){
            attSplitMeasure = new IGAttributeSplitMeasure(true);
        }else if(s.equals("chi")){
            attSplitMeasure = new ChiSquaredAttributeSplitMeasure();
        }else if(s.equals("gini")){
            attSplitMeasure = new GiniAttributeSplitMeasure();
        }else{
            throw new Exception("InvalidSplitCriterion");
        }
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        //instances
        result.setMinimumNumberInstances(2);

        return result;
    }

    /**
     * Builds a decision tree classifier.
     *
     * @param data the training data
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes() - 1) {
            throw new Exception("Class attribute must be the last index.");
        }
        //maxDepth = 2;
        //System.out.println(maxDepth);
        root = new TreeNode();
        root.buildTree(data, 0);
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     */
    @Override
    public double classifyInstance(Instance instance) {
        double[] probs = distributionForInstance(instance);

        int maxClass = 0;
        for (int n = 1; n < probs.length; n++) {
            if (probs[n] > probs[maxClass]) {
                maxClass = n;
            }
        }

        return maxClass;
    }

    /**
     * Computes class distribution for instance using the decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     */
    @Override
    public double[] distributionForInstance(Instance instance) {
        return root.distributionForInstance(instance);
    }

    /**
     * Class representing a single node in the tree.
     */
    private class TreeNode {

        /** Attribute used for splitting, if null the node is a leaf. */
        Attribute bestSplit = null;

        /** Best gain from the splitting measure if the node is not a leaf. */
        double bestGain = 0;

        /** Depth of the node in the tree. */
        int depth;

        /** The node's children if it is not a leaf. */
        TreeNode[] children;

        /** The class distribution if the node is a leaf. */
        double[] leafDistribution;

        /**
         * Recursive function for building the tree.
         * Builds a single tree node, finding the best attribute to split on using a splitting measure.
         * Splits the best attribute into multiple child tree node's if they can be made, else creates a leaf node.
         *
         * @param data Instances to build the tree node with
         * @param depth the depth of the node in the tree
         */
        void buildTree(Instances data, int depth) throws Exception {
            this.depth = depth;

            // Loop through each attribute, finding the best one.
            for (int i = 0; i < data.numAttributes() - 1; i++) {
                double gain = attSplitMeasure.computeAttributeQuality(data, data.attribute(i));

                if (gain > bestGain) {
                    bestSplit = data.attribute(i);
                    bestGain = gain;
                }
            }

            // If we found an attribute to split on, create child nodes.
            if (bestSplit != null) {
                Instances[] split;
                if (bestSplit.isNumeric()){
                    mean = data.attributeStats(bestSplit.index()).numericStats.mean;
                    split = attSplitMeasure.splitDataOnNumeric(data, bestSplit);
                }else{
                    split = attSplitMeasure.splitData(data, bestSplit);
                }
                children = new TreeNode[split.length];

                // Create a child for each value in the selected attribute, and determine whether it is a leaf or not.
                for (int i = 0; i < children.length; i++){
                    children[i] = new TreeNode();

                    boolean leaf = split[i].numDistinctValues(data.classIndex()) == 1 || depth + 1 == maxDepth;

                    if (split[i].isEmpty()) {
                        children[i].buildLeaf(data, depth + 1);
                    } else if (leaf) {
                        children[i].buildLeaf(split[i], depth + 1);
                    } else {
                        children[i].buildTree(split[i], depth + 1);
                    }
                }
            // Else turn this node into a leaf node.
            } else {
                leafDistribution = classDistribution(data);
            }
        }

        /**
         * Builds a leaf node for the tree, setting the depth and recording the class distribution of the remaining
         * instances.
         *
         * @param data remaining Instances to build the leafs class distribution
         * @param depth the depth of the node in the tree
         */
        void buildLeaf(Instances data, int depth) {
            this.depth = depth;
            leafDistribution = classDistribution(data);
        }

        /**
         * Recursive function traversing node's of the tree until a leaf is found. Returns the leafs class distribution.
         *
         * @return the class distribution of the first leaf node
         */
        double[] distributionForInstance(Instance inst) {
            // If the node is a leaf return the distribution, else select the next node based on the best attributes
            // value.
            if (bestSplit == null) {
                return leafDistribution;
            } else {
                //System.out.println(inst.value(bestSplit));
                //System.out.println(children.length);
                if (bestSplit.isNumeric()){
                    return children[(inst.value(bestSplit)>mean)?1:0].distributionForInstance(inst);
                }
                return children[(int) inst.value(bestSplit)].distributionForInstance(inst);
            }
        }

        /**
         * Returns the normalised version of the input array with values summing to 1.
         *
         * @return the class distribution as an array
         */
        double[] classDistribution(Instances data) {
            double[] distribution = new double[data.numClasses()];
            for (Instance inst : data) {
                distribution[(int) inst.classValue()]++;
            }

            double sum = 0;
            for (double d : distribution){
                sum += d;
            }

            if (sum != 0){
                for (int i = 0; i < distribution.length; i++) {
                    distribution[i] = distribution[i] / sum;
                }
            }

            return distribution;
        }

        /**
         * Summarises the tree node into a String.
         *
         * @return the summarised node as a String
         */
        @Override
        public String toString() {
            String str;
            if (bestSplit == null){
                str = "Leaf," + Arrays.toString(leafDistribution) + "," + depth;
            } else {
                str = bestSplit.name() + "," + bestGain + "," + depth;
            }
            return str;
        }
    }

    /**
     * Main method.
     *
     * @param args the options for the classifier main
     */
    public static void main(String[] args) throws Exception {
        FileReader optdigits = new FileReader("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");
        FileReader chinatown = new FileReader("src/main/java/ml_6002b_coursework/test_data/Chinatown.arff");
        Instances inst = new Instances(optdigits);
        Instances[] trainTest = Utils.splitData(inst, 0.5);
        //System.out.println("here");
        trainTest[0].setClassIndex(trainTest[0].numAttributes()-1);
        trainTest[1].setClassIndex(trainTest[1].numAttributes()-1);

        CourseworkTree tree = new CourseworkTree();


        //optdigits
        tree.setOptions(new String[]{"-split","ig"});
        tree.buildClassifier(trainTest[0]);
        double acc = ClassifierTools.accuracy(trainTest[1], tree);
        System.out.println("DT using measure Information Gain on optdigits problem has test accuracy = " + acc);

        tree.setOptions(new String[]{"-split","igratio"});
        tree.buildClassifier(trainTest[0]);
        acc = ClassifierTools.accuracy(trainTest[1], tree);
        System.out.println("DT using measure Information Gain Ratio on optdigits problem has test accuracy = " + acc);

        tree.setOptions(new String[]{"-split","chi"});
        tree.buildClassifier(trainTest[0]);
        acc = ClassifierTools.accuracy(trainTest[1], tree);
        System.out.println("DT using measure Chi on optdigits problem has test accuracy = " + acc);

        tree.setOptions(new String[]{"-split","gini"});
        tree.buildClassifier(trainTest[0]);
        acc = ClassifierTools.accuracy(trainTest[1], tree);
        System.out.println("DT using measure Gini on optdigits problem has test accuracy = " + acc);



        //Chinatown

        inst = new Instances(chinatown);
        trainTest = Utils.splitData(inst, 0.5);
        trainTest[0].setClassIndex(trainTest[0].numAttributes()-1);
        trainTest[1].setClassIndex(trainTest[1].numAttributes()-1);

        tree.setOptions(new String[]{"-split","ig"});
        tree.buildClassifier(trainTest[0]);
        acc = ClassifierTools.accuracy(trainTest[1], tree);
        System.out.println("\n\nDT using measure Information Gain on Chinatown problem has test accuracy = " + acc);

        tree.setOptions(new String[]{"-split","igratio"});
        tree.buildClassifier(trainTest[0]);
        acc = ClassifierTools.accuracy(trainTest[1], tree);
        System.out.println("DT using measure Information Gain Ratio on Chinatown problem has test accuracy = " + acc);

        tree.setOptions(new String[]{"-split","chi"});
        tree.buildClassifier(trainTest[0]);
        acc = ClassifierTools.accuracy(trainTest[1], tree);
        System.out.println("DT using measure Chi on Chinatown problem has test accuracy = " + acc);

        tree.setOptions(new String[]{"-split","gini"});
        tree.buildClassifier(trainTest[0]);
        acc = ClassifierTools.accuracy(trainTest[1], tree);
        System.out.println("DT using measure Gini on Chinatown problem has test accuracy = " + acc);

    }
}