package sk.feromakovi.backpropagation;

public class Connection {
	
	double weight = 0;
	double prevDeltaWeight = 0; // for momentum

	final Neuron leftNeuron;
	final Neuron rightNeuron;
	static int counter = 0;
	final public int id; // auto increment, starts at 0

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
