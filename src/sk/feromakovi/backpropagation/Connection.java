package sk.feromakovi.backpropagation;

import java.io.Serializable;

public class Connection implements Serializable{

	transient static int counter = 0;
	
	double weight = 0;
	double prevDeltaWeight = 0;
	final Neuron leftNeuron;
	final Neuron rightNeuron;
	final public int id;

	public Connection(Neuron fromN, Neuron toN) {
		leftNeuron = fromN;
		rightNeuron = toN;
		id = counter;
		counter++;
		this.weight = Main.generateRandom();
	}

	public double getWeight() {
		return weight;
	}

	public void setWeight(double w) {
		weight = w;
	}

	public void setDeltaWeight(double w) {
		prevDeltaWeight = w;
	}

	public double getPrevDeltaWeight() {
		return prevDeltaWeight;
	}

	public Neuron getFromNeuron() {
		return leftNeuron;
	}

	public Neuron getToNeuron() {
		return rightNeuron;
	}
}
