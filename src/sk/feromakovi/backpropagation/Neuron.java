package sk.feromakovi.backpropagation;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

public class Neuron implements Serializable{
	transient static int counter = 0;
	final public int id;
	double output;
	
	final double threshold = 1;
	double weigth;
	
	ArrayList<Connection> Inconnections = new ArrayList<Connection>();
	HashMap<Integer,Connection> connectionLookup = new HashMap<Integer,Connection>();
	
	public Neuron(){	
		this.weigth = Main.generateRandom();
		id = counter;
		counter++;
	}
	
	public void calculateOutput(){
		double s = 0;
		for(Connection con : Inconnections){
			Neuron leftNeuron = con.getFromNeuron();
			double weight = con.getWeight();
			double a = leftNeuron.getOutput();
			s += (weight * a);
		}
		s += (weigth * threshold);
		output = sigmoid(s);
	}

	double sigmoid(double x) {
		return 1.0 / (1.0 + (Math.exp(-x)));
	}
	
	public void addInConnectionsS(ArrayList<Neuron> inNeurons){
		for(Neuron n: inNeurons){
			Connection con = new Connection(n,this);
			Inconnections.add(con);
			connectionLookup.put(n.id, con);
		}
	}
	
	public Connection getConnection(int neuronIndex){
		return connectionLookup.get(neuronIndex);
	}

	public void addInConnection(Connection con){
		Inconnections.add(con);
	}
	
	public ArrayList<Connection> getAllInConnections(){
		return Inconnections;
	}
	
	public double getOutput() {
		return output;
	}
	public void setOutput(double o){
		output = o;
	}
}
