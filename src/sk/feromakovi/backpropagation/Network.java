package sk.feromakovi.backpropagation;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

public class Network {
	
	final List<Data> mData; 
	
	final boolean isTrained = false;
	final DecimalFormat df = new DecimalFormat("#.0#");
	final Random rand = new Random();
	final ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
	final ArrayList<Neuron> hiddenLayer = new ArrayList<Neuron>();
	final ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
	final int[] layers;
	final double learningRate;
	final double momentum;
	
	final HashMap<String, Double> weightUpdate = new HashMap<String, Double>();
	
	double resultOutputs[][] = { { -1 }, { -1 }, { -1 }, { -1 } ,{-1}}; // dummy init

	public Network(List<Data> data, int hiddenNeuronCount, double learningRate, double momentum) {
		this.mData = data;
		Data d = data.get(0);
		this.layers = new int[] {d.in.length, hiddenNeuronCount, d.out.length};
		this.learningRate = learningRate;
		this.momentum = momentum; 

		for(int i = 0; i < d.in.length; i++){
			Neuron neuron = new Neuron();
			inputLayer.add(neuron);
		}
		
		for(int i = 0; i < hiddenNeuronCount; i++){
			Neuron neuron = new Neuron();
			neuron.addInConnectionsS(inputLayer);
			hiddenLayer.add(neuron);
		}
		
		for(int i = 0; i < d.out.length; i++){
			Neuron neuron = new Neuron();
			neuron.addInConnectionsS(hiddenLayer);
			outputLayer.add(neuron);
		}

		// reset id counters
		Neuron.counter = 0;
		Connection.counter = 0;

		if (isTrained) {			
			updateAllWeights();
		}
	}

	
		/**
		 * 
		 * @param inputs
		 *            There is equally many neurons in the input layer as there are
		 *            in input variables
		 */
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

		/**
		 * Calculate the output of the neural network based on the input The forward
		 * operation
		 */
		public void activate() {
			for (Neuron n : hiddenLayer)
				n.calculateOutput();
			for (Neuron n : outputLayer)
				n.calculateOutput();
		}

		/**
		 * all output propagate back
		 * 
		 * @param expectedOutput
		 *            first calculate the partial derivative of the error with
		 *            respect to each of the weight leading into the output neurons
		 *            bias is also updated here
		 */
		public void applyBackpropagation(double expectedOutput[]) {
			int i = 0;
			for (Neuron n : outputLayer) {
				ArrayList<Connection> connections = n.getAllInConnections();
				double ak = n.getOutput();
				double desiredOutput = expectedOutput[i];
				double partialDerivative = ak * (1 - ak) * (desiredOutput - ak);
				for (Connection con : connections) {
					
					double ai = con.leftNeuron.getOutput();					
					double deltaWeight = (learningRate * partialDerivative * ai) + (momentum * con.getPrevDeltaWeight());
					double newWeight = con.getWeight() + deltaWeight;
					con.setDeltaWeight(deltaWeight);
					con.setWeight(newWeight);//+ momentum * con.getPrevDeltaWeight()
				}
				n.weigth += (partialDerivative * learningRate);
				i++;
			}

			// update weights for the hidden layer
			for (Neuron n : hiddenLayer) {
				ArrayList<Connection> connections = n.getAllInConnections();
				for (Connection con : connections) {
					double aj = n.getOutput();
					double ai = con.leftNeuron.getOutput();
					double sumKoutputs = 0;
					int j = 0;
					for (Neuron out_neu : outputLayer) {
						double wjk = out_neu.getConnection(n.id).getWeight();
						double desiredOutput = (double) expectedOutput[j];
						double ak = out_neu.getOutput();
						j++;
						sumKoutputs = sumKoutputs
								+ ((desiredOutput - ak) * ak * (1 - ak) * wjk);
					}

					double partialDerivative = aj * (1 - aj) * ai * sumKoutputs;
					double deltaWeight = learningRate * partialDerivative;
					double newWeight = con.getWeight() + deltaWeight;
					con.setDeltaWeight(deltaWeight);
					con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
				}
			}
		}

		void run(int maxSteps, double minError) {
			int i;
			// Train neural network until minError reached or maxSteps exceeded
			double error = 1;
			for (i = 0; i < maxSteps && error > minError; i++) {
				error = 0;
				Collections.shuffle(mData);
				for(Data data : mData){
					setInput(data.in);
					activate();
					double[] output = getOutput();
					
					error += data.getPartialError(output);
					applyBackpropagation(data.out);
				}
				error = (error)/(mData.size());
				Main.log("epoch: " + i + "   error: " + error);
				
//				for (int p = 0; p < inputs.length; p++) {
//					setInput(inputs[p]);
//
//					activate();
//
//					output = getOutput();
//					resultOutputs[p] = output;
//
//					for (int j = 0; j < expectedOutputs[p].length; j++) {
//						double err = Math.pow(output[j] - expectedOutputs[p][j], 2);
//						error += err;
//					}
//
//					applyBackpropagation(expectedOutputs[p]);
//				}
			}

			printResult();
			
			System.out.println("Sum of squared errors = " + error);
			//System.out.println("##### EPOCH " + i+"\n");
			if (i == maxSteps) {
				System.out.println("!Error training try again");
			} else {
				printAllWeights();
				printWeightUpdate();
			}
			System.out.println("pocet generacii: "+i);
		}
		
		void printResult()
		{
			System.out.println("NN example with xor training");
			for(Data d : mData){//for (int p = 0; p < inputs.length; p++) {
				System.out.print("INPUTS: ");
				for (int x = 0; x < layers[0]; x++) {
					System.out.print(d.in[x] + " ");
				}

				System.out.print("EXPECTED: ");
				for (int x = 0; x < layers[2]; x++) {
					System.out.print(d.out[x] + " ");
				}

				System.out.print("ACTUAL: ");
				for (int x = 0; x < layers[2]; x++) {
					//System.out.print(resultOutputs[p][x] + " ");
				}
				System.out.println();
			}
			System.out.println();
		}

		String weightKey(int neuronId, int conId) {
			return "N" + neuronId + "_C" + conId;
		}

		/**
		 * Take from hash table and put into all weights
		 */
		public void updateAllWeights() {
			// update weights for the output layer
			for (Neuron n : outputLayer) {
				ArrayList<Connection> connections = n.getAllInConnections();
				for (Connection con : connections) {
					String key = weightKey(n.id, con.id);
					double newWeight = weightUpdate.get(key);
					con.setWeight(newWeight);
				}
			}
			// update weights for the hidden layer
			for (Neuron n : hiddenLayer) {
				ArrayList<Connection> connections = n.getAllInConnections();
				for (Connection con : connections) {
					String key = weightKey(n.id, con.id);
					double newWeight = weightUpdate.get(key);
					con.setWeight(newWeight);
				}
			}
		}

		
		public void printWeightUpdate() {
			System.out.println("printWeightUpdate, put this i trainedWeights() and set isTrained to true");
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
}
