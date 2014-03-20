package sk.feromakovi.backpropagation;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import sk.feromakovi.backpropagation.utils.Serializers;

import com.google.common.base.Charsets;
import com.google.common.io.Files;


public class Main {
	
	static Random random = new Random();

	@Argument
    private List<String> mArguments = new ArrayList<String>();
	
	@Option(name="-data")     
    private String mDataSet;	
	
	@Option(name="-state")     
    private String mState = null;
	
	@Option(name="-rate")     
    public double mLearningRate = -1;
	
	@Option(name="-momentum")     
    public double mMomentum = -1;
	
	@Option(name="-minerror")     
    public double mMinError = 0.01;
	
	@Option(name="-e")     
    public double mEpsilon = 0.3;
	
	@Option(name="-n")     
    public int mHiddenNeurons = 2;
	
	@Option(name="-maxiter")     
    public int mMaxIter = 10000;
	
	@Option(name="-h")     
    private boolean mHelp = false;
	
	public List<Data> mData = new ArrayList<Data>();
	
	public void loadData(){
		try{
			List<String> lines = Files.readLines(new File(mDataSet), Charsets.UTF_8);
			for(String line : lines)
				mData.add(new Data(line));
		}catch(Exception e){
			e.printStackTrace();
		}	
	}
	
	public Main(String[] args){
		CmdLineParser parser = new CmdLineParser(this);
        parser.setUsageWidth(80);
        try {
            parser.parseArgument(args);                      
        } catch( CmdLineException | IllegalArgumentException e ) {
            return;
        }        
	}
	
	public void check(){
		if(mDataSet == null || mHelp || mLearningRate == -1 || mMomentum == -1)
			printHelp();
	}
	
	void printHelp(){
		System.out.println("    [-data] data set");
		System.out.println("    [-state] file saving/loading neural network state");
		System.out.println("    [-rate] learning rate");
		System.out.println("    [-momentum] momentum");
		System.out.println("    [-minerror] minimal error when program can stop");
		System.out.println("    [-maxiter] max iteration count");
		System.out.println("    [-n] count of neurons in hidden layer");
		System.out.println("    [-e] epsilon to normalise results");
		System.out.println("    [-h] help");
		System.exit(0);
	}
	
	static void log(String log){
		System.out.println(log);
	}

	public static void main(String[] args) {
		Main main = new Main(args);
		main.check();
		main.loadData();
		Network net;
		boolean test = false;
		if(main.mState != null && new File(main.mState).exists()){
			net = Serializers.loadFromFile(new File(main.mState));
			net.setData(main.mData);
			test = true;
		}else{
			net = new Network(main.mData, main.mHiddenNeurons);
		}
		
		net.setLearningRate(main.mLearningRate);
		net.setMomentum(main.mMomentum);
		net.setEpsilon(main.mEpsilon);
		if(test){
			net.calculateHit(true);
		}else{
			boolean learnt = net.run(main.mMaxIter, main.mMinError);
			if(learnt && main.mState != null && !new File(main.mState).exists())
				Serializers.saveToFile(new File(main.mState), net);
		}
	}
	
	public static double generateRandom(){
		return (random.nextDouble() - 0.5);
	}
}
