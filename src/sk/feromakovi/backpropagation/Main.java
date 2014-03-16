package sk.feromakovi.backpropagation;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import com.google.common.base.Charsets;
import com.google.common.io.Files;


public class Main {
	
	@Argument
    private List<String> mArguments = new ArrayList<String>();
	
	@Option(name="-train")     
    private String mTrain;
	
	@Option(name="-test")     
    private String mTest;
	
	@Option(name="-rate")     
    private double mLearningRate = -1;
	
	@Option(name="-momentum")     
    private double mMomentum = -1;
	
	@Option(name="-n")     
    private int mHiddenNeurons = 2;
	
	@Option(name="-h")     
    private boolean mHelp = false;
	
	private List<Data> mData = new ArrayList<Data>();
	
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
	
	public List<Data> getData(){return this.mData;}
	
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
		if(mTrain == null || mHelp || mLearningRate == -1 || mMomentum == -1)
			printHelp();
	}
	
	void printHelp(){
		System.out.println("    [-train] trenovacie data");
		System.out.println("    [-test] testovacie data");
		System.out.println("    [-rate] learning rate");
		System.out.println("    [-momentum] momentum");
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
		Network net = new Network(main.getData());
	}
}
