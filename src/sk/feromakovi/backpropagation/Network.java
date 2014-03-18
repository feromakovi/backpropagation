package sk.feromakovi.backpropagation;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Network implements Serializable {

	List<Data> mData;

	final boolean isTrained = false;
	final DecimalFormat df = new DecimalFormat("#.0#");
	final ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
	final ArrayList<Neuron> hiddenLayer = new ArrayList<Neuron>();
	final ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
	double learningRate;
	double momentum;

	public Network(List<Data> data, int hiddenNeuronCount, double learningRate,
			double momentum) {
		this.mData = data;
		Data d = data.get(0);
		this.learningRate = learningRate;
		this.momentum = momentum;

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

	public void activate() {
		for (Neuron n : hiddenLayer)
			n.calculateOutput();
		for (Neuron n : outputLayer)
			n.calculateOutput();
	}

	public void applyBackpropagation(double expectedOutput[]) {
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
				con.setWeight(newWeight);// + momentum *
											// con.getPrevDeltaWeight()
			}
			n.weigth += (partialDerivative * learningRate);
			i++;
		}

		// update weights for the hidden layer
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

				double deltaWeight = (learningRate * partialDerivative * ai)
						+ (momentum * con.getPrevDeltaWeight());
				double newWeight = con.getWeight() + deltaWeight;
				con.setDeltaWeight(deltaWeight);
				con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
			}
			n.weigth += (learningRate * partialDerivative);
		}
	}

	boolean run(int maxSteps, double minError) {
		int i;
		// Train neural network until minError reached or maxSteps exceeded
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
			error = (error) / (mData.size());
			Main.log("epoch: " + i + "   error: " + error);

		}

		printResult();

		System.out.println("Sum of squared errors = " + error);
		System.out.println("pocet generacii: " + i);
		// System.out.println("##### EPOCH " + i+"\n");
		if (i == maxSteps) {
			System.out.println("!Error training try again");
		} else {
			printAllWeights();
			printWeightUpdate();
			return true;
		}
		return false;
	}

	void printResult() {
		System.out.println("NN example with xor training");
		for (Data d : mData) {// for (int p = 0; p < inputs.length; p++) {
			System.out.print("INPUTS: ");
			for (int x = 0; x < inputLayer.size(); x++) {
				System.out.print(d.in[x] + " ");
			}

			System.out.print("EXPECTED: ");
			for (int x = 0; x < outputLayer.size(); x++) {
				System.out.print(d.out[x] + " ");
			}

			// System.out.print("ACTUAL: ");
			// for (int x = 0; x < layers[2]; x++) {
			// //System.out.print(resultOutputs[p][x] + " ");
			// }
			System.out.println();
		}
		System.out.println();
	}

	public void printWeightUpdate() {
		System.out
				.println("printWeightUpdate, put this i trainedWeights() and set isTrained to true");
		// weights for the hidden layer
		for (Neuron n : hiddenLayer) {
			ArrayList<Connection> connections = n.getAllInConnections();
			for (Connection con : connections) {
				String w = df.format(con.getWeight());
				System.out.println("weightUpdate.put(weightKey(" + n.id + ", "
						+ con.id + "), " + w + ");");
			}
		}
		// weights for the output layer
		for (Neuron n : outputLayer) {
			ArrayList<Connection> connections = n.getAllInConnections();
			for (Connection con : connections) {
				String w = df.format(con.getWeight());
				System.out.println("weightUpdate.put(weightKey(" + n.id + ", "
						+ con.id + "), " + w + ");");
			}
		}
		System.out.println();
	}

	public void printAllWeights() {
		System.out.println("printAllWeights");
		// weights for the hidden layer
		for (Neuron n : hiddenLayer) {
			ArrayList<Connection> connections = n.getAllInConnections();
			for (Connection con : connections) {
				double w = con.getWeight();
				System.out.println("n=" + n.id + " c=" + con.id + " w=" + w);
			}
		}
		// weights for the output layer
		for (Neuron n : outputLayer) {
			ArrayList<Connection> connections = n.getAllInConnections();
			for (Connection con : connections) {
				double w = con.getWeight();
				System.out.println("n=" + n.id + " c=" + con.id + " w=" + w);
			}
		}
		System.out.println();
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}
}
