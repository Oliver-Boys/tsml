package ml_6002b_coursework;

import weka.core.Instances;
import java.util.Random;

public class Utils {
    public static Instances[] splitData(Instances all, double proportionTest){
        all.randomize(new Random());

        Instances[] split = new Instances[2];
        split[0] = new Instances(all);
        split[1] = new Instances(all,0);
        //System.out.println(all.numInstances());
        for(int i = 0; i < (all.numInstances()*proportionTest); i++){
            split[1].add(split[0].get(0));
            split[0].remove(0);
            //System.out.println(split[1].size() + "," + split[0].size());
        }
        //System.out.println("done");
        return split;
    }
}
