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
	
	@Argument
    private List<String> mArguments = new ArrayList<String>();
	
	@Option(name="-train")     
    private String mTrain;
	
	@Option(name="-test")     
    private String mTest;
	
	@Option(name="-state")     
    private String mState = null;
	
	@Option(name="-rate")     
    public double mLearningRate = -1;
	
	@Option(name="-momentum")     
    public double mMomentum = -1;
	
	@Option(name="-minerror")     
    public double mMinError = 0.01;
	
	@Option(name="-n")     
    public int mHiddenNeurons = 2;
	
	@Option(name="-maxiter")     
    public int mMaxIter = 10000;
	
	@Option(name="-h")     
    private boolean mHelp = false;
	
	public List<Data> mData = new ArrayList<Data>();
	
	public void loadData(){
		try{
			List<String> lines = Files.readLines(new File(mTrain), Charsets.UTF_8);
			for(String line : lines)
				mData.add(new Data(line));
		}catch(Exception e){
			e.printStackTrace();
		}	
		//for(Data d : mData) log(d.toString());
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
		if((mTrain == null && mTest == null) || mHelp || mLearningRate == -1 || mMomentum == -1)
			printHelp();
	}
	
	void printHelp(){
		System.out.println("    [-train] trenovacie data");
		System.out.println("    [-test] testovacie data");
		System.out.println("    [-state] file saving/loading neural network state");
		System.out.println("    [-rate] learning rate");
		System.out.println("    [-momentum] momentum");
		System.out.println("    [-minerror] minimal error when program can stop");
		System.out.println("    [-maxiter] max iteration count");
		System.out.println("    [-n] pocet neuronov v skrytej vrstve");
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
		File netFile = new File("learned.dat");
		Network net;
		if(netFile.exists())
			net = Serializers.loadFromFile(netFile);
		else
			net = new Network(main.mData, main.mHiddenNeurons, main.mLearningRate, main.mMomentum);
		boolean learnt = net.run(main.mMaxIter, main.mMinError);
		if(learnt)
			Serializers.saveToFile(netFile, net);
	}
	
	static Random random = new Random();
	
	public static double generateRandom(){
		return (random.nextDouble() - 0.5);
	}
}
