package ml_6002b_coursework;

import java.util.Arrays;

/**
 * Empty class for Part 2.1 of the coursework.
 */
public class AttributeMeasures {

    /**
     * Main method.
     *
     * @param args the options for the attribute measure main
     */
    public static void main(String[] args) {
        System.out.println("Not Implemented.");

        int[][] peatyArray = new int[][]{
                {4,0},
                {1,5}
        };
        System.out.println("measure Information Gain for Peaty = " + measureInformationGain(peatyArray));
        System.out.println("measure Information Gain Ratio for Peaty = " + measureInformationGainRatio(peatyArray));
        System.out.println("measure Gini for Peaty = " + measureGini(peatyArray));
        System.out.println("measure Chi for Peaty = " + measureChiSquared(peatyArray));
    }

    public static double measureInformationGain(int[][] array){
        int count = 0;
        int[] casesAtEach = new int[array.length];
        int[] classDist = new int[array[0].length];

        double entropyRoot  = 0.0;
        double[] entropyArr = new double[array.length];
        for (int i = 0; i < array.length; i++){
            for (int j = 0; j < array[i].length; j++){
                count += array[i][j];
                casesAtEach[i] += array[i][j];
                classDist[j] += array[i][j];
            }
            for (int j = 0; j < array[i].length; j++){
                double fraction = (double) array[i][j]/casesAtEach[i];
                //System.out.println(fraction + "," + ((fraction!=0) ? ((fraction) * (Math.log10(fraction)/ Math.log10(2))) : 0));
                entropyArr[i] -= (fraction!=0) ? ((fraction) * (Math.log10(fraction)/ Math.log10(2))) : 0;
            }

        }
        for (int i = 0; i < classDist.length; i++){
            double fraction = (double) classDist[i]/count;
            entropyRoot -= ((fraction) * (Math.log10(fraction)/ Math.log10(2)));
        }
        double gain = entropyRoot;
        for (int i = 0; i < entropyArr.length; i++){
            gain -= ((double) casesAtEach[i]/count)*entropyArr[i];
        }
        return gain;
    }
    public static double measureInformationGainRatio(int[][] array){
        double gain = measureInformationGain(array);
        int count = 0;
        int[] casesAtEach = new int[array.length];
        for (int i = 0; i < array.length; i++){
            for (int j = 0; j < array[0].length; j++){
                count += array[i][j];
                casesAtEach[i] += array[i][j];
            }
        }
        double splitInfo = 0;
        for (int i = 0; i < casesAtEach.length; i++){
            double fraction = (double)casesAtEach[i]/count;
            splitInfo -= (fraction!=0) ? ((fraction) * (Math.log10(fraction)/ Math.log10(2))) : 0;
        }
        return gain/splitInfo;
    }
    public static double measureGini(int[][] array){
        int count = 0;
        int[] casesAtEach = new int[array.length];
        int[] classDist = new int[array[0].length];

        double impurityRoot = 1.0;
        double[] impurityArr = new double[array.length];
        Arrays.fill(impurityArr, 1.0);

        for (int i = 0; i < array.length; i++){
            for (int j = 0; j < array[i].length; j++){
                count += array[i][j];
                casesAtEach[i] += array[i][j];
                classDist[j] += array[i][j];
            }
            for (int j = 0; j < array[i].length; j++){
                double fraction = (double) array[i][j]/casesAtEach[i];
                //System.out.println(fraction + "," + ((fraction!=0) ? ((fraction) * (Math.log10(fraction)/ Math.log10(2))) : 0));
                impurityArr[i] -= Math.pow(fraction,2);
            }
        }
        for (int i = 0; i < classDist.length; i++){
            double fraction = (double) classDist[i]/count;
            impurityRoot -=Math.pow(fraction,2);
        }
        double gini = impurityRoot;
        for (int i = 0; i < impurityArr.length; i++){
            gini -= ((double) casesAtEach[i]/count)*impurityArr[i];
        }
        return gini;
    }
    public static double measureChiSquared(int[][] array){
        int count = 0;
        int[] casesAtEach = new int[array.length];
        int[] classDist = new int[array[0].length];

        double[][] expected = new double[array.length][array[0].length];
        for (int i = 0; i < array.length; i++){
            for (int j = 0; j < array.length; j++){
                count += array[i][j];
                casesAtEach[i] += array[i][j];
                classDist[j] += array[i][j];
            }
        }
        for (int i = 0; i < array.length; i++){
            for (int j = 0; j < array.length; j++){
                expected[i][j] = casesAtEach[i] * ((double)classDist[j]/count);
            }
        }
        double chi2 = 0;
        for (int i = 0; i < array.length; i++){
            for (int j = 0; j < array.length; j++){
                chi2 += Math.pow(array[i][j] - expected[i][j], 2)/expected[i][j];
            }
        }
        return chi2;
    }

}
