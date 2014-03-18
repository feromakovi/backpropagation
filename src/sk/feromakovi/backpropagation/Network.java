package sk.feromakovi.backpropagation;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Network implements Serializable {

	List<Data> mData;

	final boolean isTrained = false;
	final ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
	final ArrayList<Neuron> hiddenLayer = new ArrayList<Neuron>();
	final ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
	double learningRate;
	double momentum;
	double epsilon;

	public Network(List<Data> data, int hiddenNeuronCount) {
		this.mData = data;
		Data d = data.get(0);
		
		for (int i = 0; i < d.in.length; i++) {
			Neuron neuron = new Neuron();
			inputLayer.add(neuron);
		}

		for (int i = 0; i < hiddenNeuronCount; i++) {
			Neuron neuron = new Neuron();
			neuron.addInConnectionsS(inputLayer);
			hiddenLayer.add(neuron);
		}

		for (int i = 0; i < d.out.length; i++) {
			Neuron neuron = new Neuron();
			neuron.addInConnectionsS(hiddenLayer);
			outputLayer.add(neuron);
		}

		Neuron.counter = 0;
		Connection.counter = 0;
	}

	public void setInput(double inputs[]) {
		for (int i = 0; i < inputLayer.size(); i++) {
			inputLayer.get(i).setOutput(inputs[i]);
		}
	}

	public double[] getOutput() {
		double[] outputs = new double[outputLayer.size()];
		for (int i = 0; i < outputLayer.size(); i++)
			outputs[i] = outputLayer.get(i).getOutput();
		return outputs;
	}

	private void activate() {
		for (Neuron n : hiddenLayer)
			n.calculateOutput();
		for (Neuron n : outputLayer)
			n.calculateOutput();
	}

	private void applyBackpropagation(double expectedOutput[]) {
		int i = 0;
		for (Neuron n : outputLayer) {
			ArrayList<Connection> connections = n.getAllInConnections();
			double ak = n.getOutput();
			double desiredOutput = (double) expectedOutput[i];
			double partialDerivative = ak * (1 - ak) * (desiredOutput - ak);
			for (Connection con : connections) {
				double ai = con.leftNeuron.getOutput();
				double deltaWeight = (learningRate * partialDerivative * ai)
						+ (momentum * con.getPrevDeltaWeight());
				double newWeight = con.getWeight() + deltaWeight;
				con.setDeltaWeight(deltaWeight);
				con.setWeight(newWeight);
			}
			n.weigth += (partialDerivative * learningRate);
			i++;
		}

		for (Neuron n : hiddenLayer) {
			ArrayList<Connection> connections = n.getAllInConnections();

			double sumKoutputs = 0;
			int j = 0;
			for (Neuron out_neu : outputLayer) {
				double wjk = out_neu.getConnection(n.id).getWeight();
				double desiredOutput = (double) expectedOutput[j];
				double ak = out_neu.getOutput();
				j++;
				sumKoutputs += ((desiredOutput - ak) * ak * (1 - ak) * wjk);
			}
			double aj = n.getOutput();
			double partialDerivative = aj * (1 - aj) * sumKoutputs;

			for (Connection con : connections) {
				double ai = con.leftNeuron.getOutput();

				double deltaWeight = (learningRate * partialDerivative * ai) + (momentum * con.getPrevDeltaWeight());
				double newWeight = con.getWeight() + deltaWeight;
				con.setDeltaWeight(deltaWeight);
				con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
			}
			n.weigth += (learningRate * partialDerivative);
		}
	}

	public boolean run(int maxSteps, double minError) {
		int i;
		double error = 1;
		for (i = 0; i < maxSteps && error > minError; i++) {
			error = 0;
			Collections.shuffle(mData);
			for (Data data : mData) {
				setInput(data.in);
				activate();
				double[] output = getOutput();

				error += data.getPartialError(output);
				applyBackpropagation(data.out);
			}
			double hit = calculateHit();
			error = (error) / (mData.size());
			Main.log("epoch: " + i + "   data hit: " + hit + "   min squared error: " + error);
		}

		if (i == maxSteps) {
			System.out.println("!Error training try again");
		} else {
//			printAllWeights();
//			printWeightUpdate();
			return true;
		}
		return false;
	}
	
	private double calculateHit(){
		double hit = 0;
		for(Data d : mData){
			setInput(d.in);
			activate();
			double[] o = getOutput();
			int c = 0;
			for(int i = 0; i < o.length; i++){
				if(normalised(o[i]) == d.out[i])
					c++;
			}
			if(c == o.length)
				hit = hit + 1;
		}
		return ((hit / mData.size()) * 100);
	}
	
	private double normalised(double d){
		if(d <= Math.abs(0 - epsilon))
			return 0;
		else if(d >= Math.abs(1 - epsilon))
			return 1;
		return d;
	}

//	private void printResult() {
//		System.out.println("NN example with xor training");
//		for (Data d : mData) {
//			System.out.print("INPUTS: ");
//			for (int x = 0; x < inputLayer.size(); x++) {
//				System.out.print(d.in[x] + " ");
//			}
//
//			System.out.print("EXPECTED: ");
//			for (int x = 0; x < outputLayer.size(); x++) {
//				System.out.print(d.out[x] + " ");
//			}
//			System.out.println();
//		}
//		System.out.println();
//	}

//	private void printWeightUpdate() {
//		System.out
//				.println("printWeightUpdate, put this i trainedWeights() and set isTrained to true");
//		for (Neuron n : hiddenLayer) {
//			ArrayList<Connection> connections = n.getAllInConnections();
//			for (Connection con : connections) {
//				String w = df.format(con.getWeight());
//				System.out.println("weightUpdate.put(weightKey(" + n.id + ", " + con.id + "), " + w + ");");
//			}
//		}
//		for (Neuron n : outputLayer) {
//			ArrayList<Connection> connections = n.getAllInConnections();
//			for (Connection con : connections) {
//				String w = df.format(con.getWeight());
//				System.out.println("weightUpdate.put(weightKey(" + n.id + ", " + con.id + "), " + w + ");");
//			}
//		}
//		System.out.println();
//	}
//
//	private void printAllWeights() {
//		System.out.println("printAllWeights");
//		for (Neuron n : hiddenLayer) {
//			ArrayList<Connection> connections = n.getAllInConnections();
//			for (Connection con : connections) {
//				double w = con.getWeight();
//				System.out.println("n=" + n.id + " c=" + con.id + " w=" + w);
//			}
//		}
//		for (Neuron n : outputLayer) {
//			ArrayList<Connection> connections = n.getAllInConnections();
//			for (Connection con : connections) {
//				double w = con.getWeight();
//				System.out.println("n=" + n.id + " c=" + con.id + " w=" + w);
//			}
//		}
//		System.out.println();
//	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}

	public void setEpsilon(double mEpsilon) {
		this.epsilon = mEpsilon;
	}
}
