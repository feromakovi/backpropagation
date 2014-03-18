package sk.feromakovi.backpropagation.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public class Serializers {
	
	@SuppressWarnings("unchecked")
	public static <T extends Serializable> T loadFromFile(File file) {
		try {
			ObjectInputStream inputObjects = new ObjectInputStream(new FileInputStream(file));
			T mw =  (T) inputObjects.readObject();
			inputObjects.close();
			return mw;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public static <T extends Serializable> void saveToFile(File file, T page) {
		try {
			FileOutputStream output = new FileOutputStream(file);
			ObjectOutputStream outputObject = new ObjectOutputStream(output);
			outputObject.writeObject(page);
			outputObject.close();
			output.close();
		} catch (Exception e) {
			e.printStackTrace();
		} 
	}

}
