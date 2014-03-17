package sk.feromakovi.backpropagation;

public class Data {
	
	public double[] in;
	public double[] out;

	public Data(String line) {
		String[] sets = line.split("#");
		String[] ins = sets[0].split(";");
		String[] outs = sets[1].split(";");
		in = new double[ins.length];
		out = new double[outs.length];
		
		for(int i = 0; i < ins.length; i++)
			in[i] = Double.parseDouble(ins[i]);
		
		for(int i = 0; i < outs.length; i++)
			out[i] = Double.parseDouble(outs[i]);
	}
	
	public double getPartialError(double[] res){
		double error = 0;
		for (int i = 0; i < res.length; i++) {
			error += Math.pow((out[i] - res[i]), 2);
		}
		return (error / out.length);
	}

	@Override
	public String toString() {
		String r = "in: ";
		for(double d : in) r += d + " ";
		r += "    out: ";
		for(double d : out) r += d + " ";
		return r;
	}
}
