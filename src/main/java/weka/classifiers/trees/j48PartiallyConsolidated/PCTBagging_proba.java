package weka.classifiers.trees.j48PartiallyConsolidated;


import java.io.*;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.J48ItPartiallyConsolidated;
import weka.classifiers.trees.J48PartiallyConsolidated;
import weka.classifiers.trees.J48PartiallyConsolidatedParallel;
import weka.classifiers.trees.j48ItPartiallyConsolidated.C45ItPartiallyConsolidatedPruneableClassifierTree;
import weka.core.Instances;
import weka.core.Option;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

public class PCTBagging_proba {
	
	public static void PCTB_ebaluazioa(String[] v_dataset, double kontsol_ptz) throws Exception {
		
		DataSource sr = null;
		double zuzenak = 0;
		double zuzenak_bb = 0;
		int kont=0;
		long execTimeTotal = 0;
		long execTimeBB = 0;
		
		
		
		for (int i=0; i < v_dataset.length; i++) {
			try {
				sr = new DataSource(v_dataset[i]);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				break;
			}
			
			Instances ds = sr.getDataSet();
			ds.setClassIndex(ds.numAttributes()-1);
			
			//J48PartiallyConsolidatedParallel pctb_tree = new J48PartiallyConsolidatedParallel();
			J48PartiallyConsolidated pctb_tree = new J48PartiallyConsolidated();
			
			String kontsol_ptz_s = Double.toString(kontsol_ptz);
			//String num_slots_s = Integer.toString(0);
			String[] options = {"-PCTB-C",kontsol_ptz_s};//,"num-slots","2"};
			pctb_tree.setOptions(options);
			//pctb_tree.setNumExecutionSlots(2);
			//pctb_tree.setDebug(true);
			
			long startTime = System.nanoTime();
			
			//pctb_tree.buildClassifiers(ds);
			pctb_tree.buildClassifier(ds);
			
			long endTime = System.nanoTime();
			
			long execTime = (endTime - startTime) / 1000000;
			
			execTimeTotal += execTime;
			
			Evaluation eval = new Evaluation(ds);
			eval.crossValidateModel(pctb_tree, ds, 10, new Random(1));
			
			zuzenak += (eval.correct()/ds.numInstances())*100;
			kont++;
			
			System.out.println("Zuzenak (%) " + i + " garren dataset-ean: " + (eval.correct()/ds.numInstances())*100 + "\n");
		}
		
		zuzenak_bb = zuzenak/kont;
		execTimeBB = execTimeTotal/kont;
		System.out.println("Zuzenen (%) batazbestekoa " + kontsol_ptz + "ko kontsolidazio-portzentaiarekin: " + zuzenak_bb + "\n");
		System.out.println("Exekuzioak " + execTimeBB + "ms behar izan ditu \n");
	}

	public static void main(String[] args) throws Exception {
		/* Read the dataset */
		String filenames[];
		DataSource sr = null;
		filenames = new String[10];
		double time_wholect = 0;
		double time_partct = 0;
		double time_bagging = 0;
		
		//filenames[0] = "C:\\Program Files\\Weka-3-8-6\\data\\weather.nominal.arff";
		filenames[0] = "C:\\Program Files\\Weka-3-8-6\\data\\breast-cancer.arff";
		//filenames[2] = "C:\\Program Files\\Weka-3-8-6\\data\\contact-lenses.arff"; PARALELOAN ARAZOAK EMATEN DITU EZ DAKIT ZERGATIK
		//filenames[2] = "C:\\Program Files\\Weka-3-8-6\\data\\unbalanced.arff";
		//filenames[3] = "C:\\Program Files\\Weka-3-8-6\\data\\credit-g.arff";
		//filenames[4] = "C:\\Program Files\\Weka-3-8-6\\data\\diabetes.arff";
		//filenames[5] = "C:\\Program Files\\Weka-3-8-6\\data\\glass.arff";
		//filenames[6] = "C:\\Program Files\\Weka-3-8-6\\data\\hypothyroid.arff";
		//filenames[7] = "C:\\Program Files\\Weka-3-8-6\\data\\ionosphere.arff";
		//filenames[8] = "C:\\Program Files\\Weka-3-8-6\\data\\soybean.arff";
		
		//PCTB_ebaluazioa(filenames, 100.0);
		
		try {
			sr = new DataSource("C:\\Program Files\\Weka-3-8-6\\data\\hypothyroid.arff");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		Instances ds = sr.getDataSet();
		ds.setClassIndex(ds.numAttributes()-1);
		
		J48ItPartiallyConsolidated pctbit_tree = new J48ItPartiallyConsolidated();
		
		pctbit_tree.setDebug(true);
		pctbit_tree.buildClassifier(ds);
		
		time_wholect = pctbit_tree.getMeasure("measureElapsedTimeTrainingWholeCT");
		time_partct = pctbit_tree.getMeasure("measureElapsedTimeTrainingPartialCT");
		time_bagging = pctbit_tree.getMeasure("measureElapsedTimeTrainingAssocBagging");
		
		System.out.println("Time Whole CT: " + time_wholect + "\n");
		System.out.println("Time Partial CT: " + time_partct + "\n");
		System.out.println("Time Bagging: " + time_bagging + "\n");
		
		
		//String filename = "C:\\Program Files\\Weka-3-8-6\\data\\breast-cancer.arff";
		//DataSource source = null;
		//try {
			//source = new DataSource(filename);
		//} catch (Exception e) {
			// TODO Auto-generated catch block
			//e.printStackTrace();
		//}
		//Load the instances
		//Instances dataset = source.getDataSet();
		//Set the class index to the class-attribute
		//dataset.setClassIndex(dataset.numAttributes()-1);
		//Specify the parameters for J48
		//String[] options = {"-C","0.25","-M","30"};
		//Build the classifier and train it
		//J48PartiallyConsolidated pctb_tree = new J48PartiallyConsolidated();
		//Vector<Option> options = new Vector<Option>();
		//Enumeration<Option> list_options = pctb_tree.listOptions();
		//while (list_options.hasMoreElements()) {
			//System.out.println(list_options.nextElement().name() + " " + list_options.nextElement().description());
		//}
		//System.out.println("\n ================================================================================ \n");
		//Aldatu defektuzko %20 kontsolidazio portzentaia %80ra
		//String[] options = {"-PCTB-C","0"};
		//pctb_tree.setOptions(options);
		
		//Uneko konfigurazioa erakutsi
		//String[] possible_options = pctb_tree.getOptions();
		//for (int i=0; i<possible_options.length; i++) System.out.println(possible_options[i]);
		
		//Klasifikatzailea eraiki
		//pctb_tree.buildClassifier(dataset);
		//Perform the evaluation
		//Evaluation eval = new Evaluation(dataset);
		//eval.crossValidateModel(pctb_tree, dataset, 10, new Random(1));
		//Print the evaluation
		//System.out.println(eval.toSummaryString("\n /***** J48PartiallyConsolidated Algorithm *****/ \n", true));

	}

}
