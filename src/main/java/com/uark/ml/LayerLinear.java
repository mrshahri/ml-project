package com.uark.ml;

/**
 * Created by rakib on 1/22/2018.
 */
public class LayerLinear extends Layer {
    LayerLinear(int inputs, int outputs) {
        super(inputs, outputs);
    }

    void activate(Vec weights, Vec x) {

        // re-initialize activation
        activation.fill(0);

        // get b
        Vec b = new Vec(weights, 0, outputs);

        // calculate Mx
        Vec Mx = new Vec(outputs);
        for (int i=1; i <= outputs; ++i) {
            Vec mRows = new Vec(weights, i*inputs-1, inputs);
            Mx.set(i-1, x.dotProduct(mRows));
        }

        // calculate activation=Mx+b
        activation.add(Mx);
        activation.add(b);
    }

    public static void main(String []args) {
        System.out.println("Testing activation function");
        Vec x = new Vec(new double[]{0, 1, 2});
        Vec weights = new Vec(new double[] {1, 5, 1, 2, 3, 2, 1, 0});
        LayerLinear layerLinear = new LayerLinear(3, 2);
        layerLinear.activate(weights, x);
        System.out.println(layerLinear.activation.toString());
    }
}
